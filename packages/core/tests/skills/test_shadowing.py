"""SK-7: a project skill shadows a user skill of the same name, and the index shows the
shadowed one as shadowed. A knowledge layer that silently replaces knowledge is worse than
none."""

from __future__ import annotations

from pathlib import Path

from skills_helpers import Tree, write_skill

from chickadee.skills.discover import discover
from chickadee.skills.index import render_index


def test_project_wins_and_the_user_skill_is_listed_as_shadowed(tmp_path: Path) -> None:
    tree = Tree(tmp_path)
    write_skill(tree.user, "migrations", "mine, across repositories")
    write_skill(tree.project, "migrations", "this project's")
    catalog = discover(tree.roots())
    assert [(s.name, s.scope) for s in catalog.skills] == [("migrations", "project")]
    (shadow,) = catalog.shadowed
    assert (shadow.skill.scope, shadow.by.scope) == ("user", "project")
    index = render_index(catalog)
    assert "- migrations · project · this project's" in index
    assert "- migrations · user · shadowed by the project skill of the same name" in index


def test_different_names_shadow_nothing(tmp_path: Path) -> None:
    tree = Tree(tmp_path)
    write_skill(tree.user, "style")
    write_skill(tree.project, "migrations")
    catalog = discover(tree.roots())
    assert {s.name for s in catalog.skills} == {"style", "migrations"}
    assert catalog.shadowed == ()
    assert "shadowed" not in render_index(catalog)
