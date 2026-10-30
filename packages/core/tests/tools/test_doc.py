"""`work.py`'s pure checks (B4): `work.dir`'s shape, a folder's
name, whose a `work.dir` is, and what a `doc` file may name. The calls themselves are
`engine/test_work_folder.py`'s (Zeus) and `box/test_plan_docs.py`'s (Athena)."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from chickadee.config import WorkPolicy
from chickadee.handoffs import SHIPPED
from chickadee.work import (
    MARKER,
    design_errors,
    dir_error,
    file_error,
    folder_name,
    in_work,
    marked,
    owned_error,
    plan_errors,
    plan_tasks,
    sections,
    unfilled,
)


@pytest.mark.parametrize("value", ["docs/chickadee", "notes", "a/b/c"])
def test_a_work_dir_inside_the_repository_is_taken(value: str) -> None:
    assert dir_error(value) is None
    assert WorkPolicy(dir=value).dir == value


@pytest.mark.parametrize("value", ["/abs", "../up", "a/../b", ".", "~/x", ".git/x", "a\\b"])
def test_a_work_dir_that_leaves_the_repository_is_refused_at_load(value: str) -> None:
    assert dir_error(value) is not None
    with pytest.raises(ValueError, match=r"work\.dir"):
        WorkPolicy(dir=value)


def test_a_folder_is_dated_slugged_and_never_one_that_exists() -> None:
    today = date(2026, 10, 18)
    taken = {"2026-10-18-add-a-retry", "2026-10-18-add-a-retry-2"}
    assert folder_name(today, "Add a retry!", lambda n: False) == "2026-10-18-add-a-retry"
    assert folder_name(today, "Add a retry", taken.__contains__) == "2026-10-18-add-a-retry-3"
    assert folder_name(today, "???", lambda n: False) == "2026-10-18-work"


def test_a_work_dir_is_refused_only_when_it_is_not_a_folder(tmp_path: Path) -> None:
    """D230 Amendment 1. Was: refused while any file directly in it lacked the marker."""
    assert owned_error(tmp_path, "docs/chickadee") is None
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "chickadee").write_text("a file, not a folder\n")
    why = owned_error(tmp_path, "docs/chickadee")
    assert why is not None and "not a folder" in why and "[work] dir" in why


@pytest.mark.parametrize(
    "theirs",
    ["notes.md", "README.md", "old/README.md", ".DS_Store", "Thumbs.db", "drafts/idea.md"],
)
def test_the_persons_own_files_in_the_work_dir_never_block_it(
    tmp_path: Path, theirs: str
) -> None:
    """The harness writes only folders it makes, under names nothing there
     has, so a file or folder of the person's beside them is never a reason to refuse. Was
    : refused, naming the file, the person to edit `[work] dir`."""
    path = tmp_path / "docs" / "chickadee" / theirs
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("mine\n")
    assert owned_error(tmp_path, "docs/chickadee") is None


def test_marked_is_the_first_line_only(tmp_path: Path) -> None:
    ours, theirs, prose = tmp_path / "a.md", tmp_path / "b.md", tmp_path / "c.md"
    ours.write_text(f"{MARKER}\n# x\n")
    theirs.write_text("# mine\n")
    prose.write_text(f"# mine\n{MARKER}\n")
    assert marked(ours) and not marked(theirs) and not marked(prose)
    assert not marked(tmp_path / "absent.md")


@pytest.mark.parametrize("file", ["README.md", "tasks/T1-x.md", "a/b.md"])
def test_a_file_in_the_folder_is_taken(file: str) -> None:
    assert file_error(file) is None


@pytest.mark.parametrize("file", ["", "/etc/passwd", "../x.md", "a/../../b", "a\\b"])
def test_a_file_that_leaves_the_folder_is_refused(file: str) -> None:
    assert file_error(file) is not None


def test_in_work_is_by_whole_path_parts() -> None:
    assert in_work("docs/chickadee/x/design/a.md", "docs/chickadee")
    assert in_work("docs/chickadee", "docs/chickadee")
    assert not in_work("docs/chickadee-old/a.md", "docs/chickadee")
    assert not in_work("docs/a.md", "docs/chickadee")


# -- the shape (Amendment 1) -----------------------------------------------------------------


DESIGN_T = (SHIPPED / "design.md").read_text(encoding="utf-8")
PLAN_T = (SHIPPED / "plan.md").read_text(encoding="utf-8")
BRIEF_T = (SHIPPED / "brief.md").read_text(encoding="utf-8")

DESIGN_OK = """# Add a retry

## Request
add a retry

## Goal
A failed call is tried again, three times.

## Extra thoughts
An extra section is the model's to add (A2).

## What changes
- `client.py`: `call()` retries.

## What must not change
- `call()`'s signature.

## Behaviour
- Three failures raise the last error.

## Outside the repository
None.

## Constraints from our conversation
None bind.

## Files
- [api.md](api.md): the retry's knobs.
"""

BRIEF_OK = """# T1: Retry the call

## Goal
`call()` retries three times.

## Where to look
- `client.py`: `call()`.

## Approach
A loop.

## Don't
- Edit a test to pass.

## Done when
`pytest -q` passes.

## Report
What changed.
"""

PLAN_OK = """# Plan: a retry

## Tasks
1. [T1: Retry the call](tasks/T1-retry.md): the loop, with tests. Depends on: none

## Final check
The full suite.
"""


def _design(
    tmp_path: Path, readme: str = DESIGN_OK, *, extra: dict[str, str] | None = None
) -> Path:
    folder = tmp_path / "design"
    folder.mkdir()
    (folder / "README.md").write_text(readme)
    for name, text in (extra if extra is not None else {"api.md": "# API\n"}).items():
        (folder / name).write_text(text)
    return folder


def _plan(tmp_path: Path, readme: str = PLAN_OK, tasks: dict[str, str] | None = None) -> Path:
    folder = tmp_path / "plan"
    (folder / "tasks").mkdir(parents=True)
    (folder / "README.md").write_text(readme)
    for name, text in (tasks if tasks is not None else {"T1-retry.md": BRIEF_OK}).items():
        (folder / "tasks" / name).write_text(text)
    return folder


def test_a_design_with_every_section_and_its_files_passes(tmp_path: Path) -> None:
    assert design_errors(_design(tmp_path), DESIGN_T) == []


def test_the_template_itself_is_not_a_filled_design(tmp_path: Path) -> None:
    errors = design_errors(_design(tmp_path, DESIGN_T, extra={}), DESIGN_T)
    assert "design/README.md: the Goal section is not filled in" in errors
    assert "design/README.md: the What changes section is not filled in" in errors


def test_a_missing_section_is_named(tmp_path: Path) -> None:
    errors = design_errors(_design(tmp_path, DESIGN_OK.replace("## Goal", "## Aim")), DESIGN_T)
    assert errors == ["design/README.md: the Goal section is missing"]


def test_files_must_match_the_folder_both_ways(tmp_path: Path) -> None:
    errors = design_errors(_design(tmp_path, extra={"db.md": "# DB\n"}), DESIGN_T)
    assert errors == [
        "design/README.md Files lists api.md, which design/ does not hold",
        "design/db.md is not listed under Files in design/README.md",
    ]


def test_a_files_line_out_of_form_is_named(tmp_path: Path) -> None:
    readme = DESIGN_OK.replace("- [api.md](api.md): the retry's knobs.", "- api.md")
    errors = design_errors(_design(tmp_path, readme), DESIGN_T)
    assert errors[0].startswith("design/README.md Files: '- api.md' is not")


def test_none_fills_an_empty_files_section(tmp_path: Path) -> None:
    readme = DESIGN_OK.replace("- [api.md](api.md): the retry's knobs.", "None.")
    assert design_errors(_design(tmp_path, readme, extra={}), DESIGN_T) == []


def test_a_plan_with_its_task_files_passes_and_its_tasks_are_read(tmp_path: Path) -> None:
    assert plan_errors(_plan(tmp_path), PLAN_T, BRIEF_T) == []
    assert plan_tasks(PLAN_OK) == (("T1", "Retry the call", "tasks/T1-retry.md"),)


def test_a_listed_task_without_its_file_and_an_unlisted_file_are_named(tmp_path: Path) -> None:
    errors = plan_errors(_plan(tmp_path, tasks={"T2-other.md": BRIEF_OK}), PLAN_T, BRIEF_T)
    assert errors == [
        "plan/tasks/T1-retry.md does not exist, though Tasks lists T1",
        "plan/tasks/T2-other.md is not listed under Tasks in plan/README.md",
    ]


def test_a_task_file_titled_otherwise_or_missing_a_section_is_named(tmp_path: Path) -> None:
    brief = BRIEF_OK.replace("# T1: Retry the call", "# T1: Something else").replace(
        "## Done when\n`pytest -q` passes.\n", ""
    )
    errors = plan_errors(_plan(tmp_path, tasks={"T1-retry.md": brief}), PLAN_T, BRIEF_T)
    assert errors == [
        "plan/tasks/T1-retry.md: its title is not `# T1: Retry the call`, its line's",
        "plan/tasks/T1-retry.md: the Done when section is missing",
    ]


def test_tasks_out_of_order_or_depending_forward_are_named(tmp_path: Path) -> None:
    readme = PLAN_OK.replace(
        "1. [T1: Retry the call](tasks/T1-retry.md): the loop, with tests. Depends on: none",
        "1. [T1: Retry the call](tasks/T1-retry.md): the loop, with tests. Depends on: T2",
    )
    errors = plan_errors(_plan(tmp_path, readme), PLAN_T, BRIEF_T)
    assert errors == ["plan/README.md Tasks: T1 depends on T2, which is not an earlier task"]


def test_a_section_of_fixed_text_copied_as_written_is_filled() -> None:
    """The brief's `Report` is the worker's instruction, with no `<…>` placeholder: copied as
    written, it is filled. Counted as unfilled, only rewording it passed (2026-10-23)."""
    report = sections(BRIEF_T)["report"]
    copied = BRIEF_OK.replace("## Report\nWhat changed.\n", f"## Report\n{report}")
    assert copied != BRIEF_OK
    assert unfilled(copied, BRIEF_T) == []


def test_a_missing_section_of_fixed_text_says_to_copy_it() -> None:
    without = BRIEF_OK.split("## Report")[0]
    assert unfilled(without, BRIEF_T) == [
        "the Report section is missing: copy it from the template as it is"
    ]
    assert unfilled(
        BRIEF_OK.replace("## Goal\n`call()` retries three times.\n", ""), BRIEF_T
    ) == ["the Goal section is missing"]


def test_unfilled_never_checks_an_extra_section() -> None:
    assert unfilled(DESIGN_OK, DESIGN_T) == []


# -- the review of #607 ----------------------------------------------------------------------


def test_in_work_folds_case_as_a_case_insensitive_disk_does() -> None:
    assert in_work("Docs/Chickadee/x/design/a.md", "docs/chickadee")
    assert in_work("docs/chickadee/a.md", "DOCS/chickadee")


@pytest.mark.parametrize("link", ["tasks/T1-../../../../x.md", "tasks/T1-x/../../../y.md"])
def test_a_task_link_that_leaves_plan_is_named_and_never_read(
    tmp_path: Path, link: str
) -> None:
    (tmp_path / "x.md").write_text(BRIEF_OK)
    readme = PLAN_OK.replace("tasks/T1-retry.md", link)
    errors = plan_errors(_plan(tmp_path, readme=readme, tasks={}), PLAN_T, BRIEF_T)
    assert f"plan/README.md Tasks: T1's file {link} leaves plan/" in errors
