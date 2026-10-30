"""An unknown name returns near-matches and loads nothing, never a guess at which file
was meant."""

from __future__ import annotations

from pathlib import Path

from skills_helpers import Tree, new_session, write_skill

from chickadee.context.results import InvalidArgument
from chickadee.skills.discover import discover
from chickadee.skills.load import NoSkills, Skills


def test_a_near_miss_offers_the_near_names_and_loads_nothing(tmp_path: Path) -> None:
    tree = Tree(tmp_path)
    write_skill(tree.project, "migrations")
    write_skill(tree.project, "style")
    session = new_session(tmp_path)
    try:
        got = Skills(discover(tree.roots()), session).load("migration", "k1")
        assert isinstance(got, InvalidArgument) and got.field == "name"
        assert "did you mean migrations?" in got.why
        assert session.state.last_seq == 0
    finally:
        session.close()


def test_a_name_near_nothing_points_at_the_index(tmp_path: Path) -> None:
    session = new_session(tmp_path)
    try:
        tree = Tree(tmp_path)
        write_skill(tree.project, "style")
        got = Skills(discover(tree.roots()), session).load("kubernetes", "k1")
        assert isinstance(got, InvalidArgument) and "the index lists them all" in got.why
    finally:
        session.close()


def test_a_session_with_no_skills_says_so() -> None:
    got = NoSkills().load("anything", "k1")
    assert isinstance(got, InvalidArgument) and "this session has none" in got.why
