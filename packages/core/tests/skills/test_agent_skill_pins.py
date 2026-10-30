"""T-3.188's pins: what the skills layer answers in older versions changes it. Each test names
the answer that moves; the task's Log lists every move and why it is right."""

from __future__ import annotations

from pathlib import Path

import pytest
from skills_helpers import Tree, write_skill

from chickadee.errors import ConfigError
from chickadee.skills.discover import discover
from chickadee.skills.index import render_index


@pytest.mark.parametrize(
    "front",
    ["license: MIT\n", "metadata:\n  author: me\n", "context: fork\n"],
    ids=["license", "metadata", "context"],
)
def test_pin_1_moved_our_skill_with_a_standard_or_agent_field_loads(
    tmp_path: Path, front: str
) -> None:
    """Moved: was refused, `front matter may not declare`. These are the
    standard's fields and Claude Code's `context`, which a skill written here may now carry."""
    tree = Tree(tmp_path)
    write_skill(tree.project, "deploy", front=front)
    (skill,) = discover(tree.roots()).skills
    assert skill.name == "deploy" and skill.ignored == ()


def test_pin_2_moved_an_external_fork_skill_is_read_as_an_agent_skill(tmp_path: Path) -> None:
    """Moved: was a knowledge skill with `context` and `agent` ignored. They
    are now read, and nothing is ignored."""
    tree = Tree(tmp_path)
    front = "context: fork\nagent: Explore\n"
    write_skill(tree.external(), "survey", "survey the code", front=front, bundle=True)
    (skill,) = discover(tree.roots()).skills
    assert skill.ignored == ()
    assert skill.front.is_agent and skill.front.agent == "Explore"


def test_pin_3_moved_past_24_the_index_names_find_skill(tmp_path: Path) -> None:
    """Moved: past the cap it said "Ask the person for the rest", and there
    was no search. It now names `find_skill`. The 24 rows are unchanged."""
    tree = Tree(tmp_path)
    for n in range(30):
        write_skill(tree.project, f"s{n:02d}")
    index = render_index(discover(tree.roots()))
    assert len([x for x in index.splitlines() if x.startswith("- ")]) == 24
    withheld = "[24 of 30 skills shown; 6 withheld (6 project). find_skill finds them"
    assert withheld in index
    assert "spawn_agent" not in index, "a catalog with no agent skill says nothing of agents"


@pytest.mark.parametrize(
    ("front", "named"),
    [("model: opus\n", "`model`"), ("allowed-tools: Read\n", "`allowed-tools`")],
)
def test_pin_4_held_model_and_allowed_tools_on_a_knowledge_skill_are_refused_by_name(
    tmp_path: Path, front: str, named: str
) -> None:
    tree = Tree(tmp_path)
    write_skill(tree.project, "deploy", front=front)
    with pytest.raises(ConfigError, match=named):
        discover(tree.roots())


def test_pin_6_moved_an_agent_file_is_an_agent_skill(tmp_path: Path) -> None:
    """Moved: was not read at all. It is now an agent skill on
    `reader`, its tools narrowed to ours."""
    tree = Tree(tmp_path)
    agents = tree.repo / ".claude" / "agents"
    agents.mkdir(parents=True)
    (agents / "reviewer.md").write_text(
        "---\nname: reviewer\ndescription: reviews code\ntools: Read, Grep\n---\nReview.\n",
        encoding="utf-8",
    )
    (skill,) = discover(tree.roots()).skills
    assert (skill.name, skill.kind, skill.profile) == ("reviewer", "agent", "reader")
    assert skill.tools == ("read", "grep")
