"""SK-2: more skills than `skills.index_max_rows` renders the cap AND states the withheld count
by scope; a description past `skills.index_desc_chars` is cut visibly. Never a silent
truncation (invariants 16 and 20). Both are rows of the limits register."""

from __future__ import annotations

from pathlib import Path

from skills_helpers import Tree, write_skill

from chickadee.limits import limit
from chickadee.skills.discover import discover
from chickadee.skills.index import render_index

SKILL_INDEX_MAX = int(limit("skills.index_max_rows").value)
SKILL_DESC_MAX = int(limit("skills.index_desc_chars").value)


def test_past_the_cap_the_withheld_count_is_stated_by_scope(tmp_path: Path) -> None:
    tree = Tree(tmp_path)
    for n in range(SKILL_INDEX_MAX):
        write_skill(tree.project, f"p{n:02d}")
    for n in range(4):
        write_skill(tree.user, f"u{n}")
    write_skill(tree.external(), "e0", bundle=True)
    index = render_index(discover(tree.roots()))
    rows = [line for line in index.splitlines() if line.startswith("- ")]
    assert len(rows) == SKILL_INDEX_MAX
    total = SKILL_INDEX_MAX + 5
    assert index.splitlines()[-1] == (
        f"[{SKILL_INDEX_MAX} of {total} skills shown; 5 withheld "
        "(1 external project, 4 user). find_skill finds them by what they are for.]"
    )


def test_at_or_under_the_cap_nothing_is_said_about_withholding(tmp_path: Path) -> None:
    tree = Tree(tmp_path)
    write_skill(tree.project, "one")
    assert "withheld" not in render_index(discover(tree.roots()))


def test_a_long_description_is_cut_visibly_to_the_cap(tmp_path: Path) -> None:
    tree = Tree(tmp_path)
    write_skill(tree.project, "long", "word " * 200)
    (row,) = [
        x for x in render_index(discover(tree.roots())).splitlines() if x.startswith("- ")
    ]
    description = row.split(" · ", 2)[2]
    assert len(description) == SKILL_DESC_MAX and description.endswith("…")


def test_no_skills_is_an_empty_index(tmp_path: Path) -> None:
    assert render_index(discover(Tree(tmp_path).roots())) == ""


def test_the_config_rows_set_both_cuts(tmp_path: Path) -> None:
    """An override in `[limits]` is what the engine passes: the index obeys it."""
    tree = Tree(tmp_path)
    for n in range(5):
        write_skill(tree.project, f"p{n}", "word " * 20)
    index = render_index(discover(tree.roots()), max_rows=2, desc_chars=10)
    rows = [line for line in index.splitlines() if line.startswith("- ")]
    assert len(rows) == 2 and all(r.split(" · ", 2)[2] == "word word…" for r in rows)
    assert "[2 of 5 skills shown; 3 withheld (3 project)." in index
