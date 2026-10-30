"""SK-16: a skill loaded and unloaded before any cut needs no special path. At the next cut its
tool line is not in the pinned set, so it is collapsed like any other result."""

from __future__ import annotations

from pathlib import Path

from skills_helpers import Tree, context_text, filler, load, must_fit, new_session, write_skill

from chickadee.skills.discover import discover
from chickadee.skills.load import Skills
from chickadee.skills.pin import unload_skill

BODY = "# Style\n\nPrefer small functions; name them for what they return.\n"


def test_an_unloaded_body_collapses_like_any_result(tmp_path: Path) -> None:
    tree = Tree(tmp_path)
    write_skill(tree.project, "style", body=BODY)
    session = new_session(tmp_path)
    try:
        filler(session)
        load(session, Skills(discover(tree.roots()), session), "style", "k1")
        assert BODY.strip() in context_text(session), "in context until the cut"
        unload_skill(session, "style")
        filler(session)
        done = must_fit(session)
        assert done.cut is not None and done.cut.lifted == ()
        assert BODY.strip() not in context_text(session)
    finally:
        session.close()
