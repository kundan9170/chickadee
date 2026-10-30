"""SK-11, SK-17: exceeding `SKILL_PINNED_MAX` REFUSES the new load and names what is pinned. It
never evicts one to make room."""

from __future__ import annotations

from pathlib import Path

from skills_helpers import Tree, load, new_session, write_skill

from chickadee.context.results import InvalidArgument
from chickadee.skills.discover import discover
from chickadee.skills.load import Skills
from chickadee.skills.pin import SKILL_PINNED_MAX, pinned


def test_the_next_load_past_the_cap_is_refused_and_the_pinned_set_is_untouched(
    tmp_path: Path,
) -> None:
    tree = Tree(tmp_path)
    names = [f"s{n}" for n in range(SKILL_PINNED_MAX + 1)]
    for name in names:
        write_skill(tree.project, name)
    session = new_session(tmp_path)
    try:
        skills = Skills(discover(tree.roots()), session)
        for n, name in enumerate(names[:-1]):
            load(session, skills, name, f"k{n}")
        got = skills.load(names[-1], "k-over")
        assert isinstance(got, InvalidArgument)
        assert all(name in got.why for name in names[:-1]), "names the pinned set"
        assert "only the person can /unload" in got.why
        assert [p.name for p in pinned(session)] == names[:-1], "nothing evicted"
    finally:
        session.close()


def test_loading_a_pinned_skill_again_is_not_a_second_pin(tmp_path: Path) -> None:
    tree = Tree(tmp_path)
    write_skill(tree.project, "style")
    session = new_session(tmp_path)
    try:
        skills = Skills(discover(tree.roots()), session)
        load(session, skills, "style", "k1")
        again = skills.load("style", "k2")
        assert again == "style is already loaded and pinned for this session."
        assert [(p.name, p.call_id) for p in pinned(session)] == [("style", "k1")]
    finally:
        session.close()
