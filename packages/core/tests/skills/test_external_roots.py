"""The compatibility roots. A `.claude/skills/` skill indexes as `external`, is
shadowed visibly by a `.chickadee/` one, has `allowed-tools` ignored with a note rather than
refused, and disappears under `skills.external = false`. Nothing of ours writes there."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from skills_helpers import Tree, load, new_session, write_skill

from chickadee.config import SkillsPolicy
from chickadee.errors import ConfigError
from chickadee.skills.discover import discover
from chickadee.skills.index import render_index
from chickadee.skills.load import Skills


@pytest.mark.parametrize(("tool", "home"), [(".claude", False), (".agents", True)])
def test_an_external_skill_indexes_as_external(tmp_path: Path, tool: str, home: bool) -> None:
    tree = Tree(tmp_path)
    write_skill(tree.external(tool, home=home), "pdf-forms", "fill PDF forms", bundle=True)
    (skill,) = discover(tree.roots()).skills
    assert (skill.name, skill.scope, skill.origin) == (
        "pdf-forms",
        "external",
        "user" if home else "project",
    )
    where = "external user" if home else "external project"
    assert f"- pdf-forms · {where} · fill PDF forms" in render_index(discover(tree.roots()))


def test_ours_shadows_an_external_one_and_the_index_says_so(tmp_path: Path) -> None:
    tree = Tree(tmp_path)
    write_skill(tree.external(), "migrations", "theirs", bundle=True)
    write_skill(tree.project, "migrations", "ours")
    catalog = discover(tree.roots())
    (winner,) = catalog.skills
    assert winner.scope == "project" and winner.front.description == "ours"
    index = render_index(catalog)
    assert "- migrations · external project · shadowed by the project skill" in index


def test_allowed_tools_and_model_are_ignored_with_a_note_not_refused(tmp_path: Path) -> None:
    """On a knowledge skill `allowed-tools` means nothing here, so it is ignored; `metadata` is
    the standard's and is read."""
    tree = Tree(tmp_path)
    front = "allowed-tools: Bash(git:*)\nmodel: opus\nmetadata:\n  version: 2\n"
    write_skill(tree.external(), "commit", "write commits", front=front, bundle=True)
    (skill,) = discover(tree.roots()).skills
    assert skill.ignored == ("allowed-tools", "model")
    assert skill.front.meta("version") == "2"
    assert "(ignored from another tool's format: allowed-tools, model)" in (
        render_index(discover(tree.roots()))
    )


def test_external_false_turns_the_roots_off(tmp_path: Path) -> None:
    tree = Tree(tmp_path)
    write_skill(tree.external(), "commit", bundle=True)
    write_skill(tree.external(".agents", home=True), "review", bundle=True)
    assert len(discover(tree.roots()).skills) == 2
    assert discover(tree.roots(external=False)).skills == ()
    assert SkillsPolicy().external is True, "on unless the person turns it off"


def test_nothing_of_ours_writes_under_an_external_root(tmp_path: Path) -> None:
    tree = Tree(tmp_path)
    path = write_skill(tree.external(), "commit", bundle=True)
    before = {
        p: (p.stat().st_mtime_ns, p.read_bytes())
        for p in tree.external().rglob("*")
        if p.is_file()
    }
    os.chmod(path, 0o444)
    session = new_session(tmp_path)
    try:
        load(session, Skills(discover(tree.roots()), session), "commit", "k1")
    finally:
        session.close()
    after = {
        p: (p.stat().st_mtime_ns, p.read_bytes())
        for p in tree.external().rglob("*")
        if p.is_file()
    }
    assert after == before


@pytest.mark.parametrize(
    "text",
    ["no front matter at all\n", "---\nname: Has Spaces\ndescription: d\n---\n"],
    ids=["no-front-matter", "unusable-name"],
)
def test_a_malformed_external_file_is_skipped_and_named_not_refused(
    tmp_path: Path, text: str
) -> None:
    """One broken file another tool wrote must not stop a session."""
    tree = Tree(tmp_path)
    write_skill(tree.external(), "good", "a usable one", bundle=True)
    broken = tree.external() / "broken" / "SKILL.md"
    broken.parent.mkdir(parents=True)
    broken.write_text(text, encoding="utf-8")
    catalog = discover(tree.roots())
    assert [s.name for s in catalog.skills] == ["good"]
    (skipped,) = catalog.skipped
    assert skipped.path == broken
    assert f"[skipped, not a skill we can read: {broken} (" in render_index(catalog)


def test_an_external_skill_named_for_a_built_in_still_refuses(tmp_path: Path) -> None:
    """SK-18 is every scope's: `/status` must mean one thing, whoever wrote the file."""
    tree = Tree(tmp_path)
    write_skill(tree.external(), "status", bundle=True)
    with pytest.raises(ConfigError, match="/status"):
        discover(tree.roots())
