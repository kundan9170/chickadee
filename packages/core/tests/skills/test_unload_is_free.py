"""SK-16: unload performs no prefix rewrite of its own. The slot frees at once; the bytes leave
at the next cut, the one that was coming anyway. So unload-then-load costs one rebuild, not
two."""

from __future__ import annotations

from pathlib import Path

from skills_helpers import (
    Tree,
    compaction,
    context_text,
    filler,
    load,
    must_fit,
    new_session,
    write_skill,
)

from chickadee.context.request import assemble
from chickadee.skills.discover import discover
from chickadee.skills.load import Skills
from chickadee.skills.pin import SKILL_PINNED_MAX, pinned, unload_skill


def test_unload_changes_nothing_sent_until_the_next_cut_and_frees_the_slot_now(
    tmp_path: Path,
) -> None:
    tree = Tree(tmp_path)
    names = [f"s{n}" for n in range(SKILL_PINNED_MAX + 1)]
    for name in names:
        write_skill(tree.project, name, body=f"# {name}\n\nthe {name} rule, at length.\n")
    session = new_session(tmp_path)
    driver = compaction()
    try:
        skills = Skills(discover(tree.roots()), session)
        filler(session)
        for n, name in enumerate(names[:-1]):
            load(session, skills, name, f"k{n}")
        filler(session)
        assert must_fit(session, driver).cut is not None
        sent = assemble(session.state, session.blob).messages

        unload_skill(session, "s0")
        assert assemble(session.state, session.blob).messages == sent, "no rewrite of its own"
        assert "s0" not in {p.name for p in pinned(session)}

        last = load(session, skills, names[-1], "k-last")
        assert "the s4 rule" in last, "the freed slot admits the next load at once"

        filler(session)
        assert must_fit(session, driver).cut is not None
        text = context_text(session)
        assert "the s0 rule" not in text, "gone at the next cut"
        assert all(f"the {n} rule" in text for n in names[1:])
    finally:
        session.close()
