"""§5.3, §9.1 authorship: an invocation's arguments render as the user's words beside a body
that is its author's, never merged into one thing."""

from __future__ import annotations

from pathlib import Path

from skills_helpers import Tree, new_session, write_skill

from chickadee.skills.discover import discover
from chickadee.skills.invoke import Invoked, invoke
from chickadee.skills.load import Skills


def test_the_arguments_are_a_block_of_their_own(tmp_path: Path) -> None:
    tree = Tree(tmp_path)
    write_skill(tree.project, "migrations", body="# M\n\nname the table you migrate\n")
    session = new_session(tmp_path)
    try:
        got = invoke(
            Skills(discover(tree.roots()), session), "migrations", " events ", call_id="u"
        )
        assert isinstance(got, Invoked)
        assert got.arguments == "<user_arguments>\nevents\n</user_arguments>"
        assert "events" not in got.body
        text = got.render()
        assert text.index("</skill>") < text.index("<user_arguments>")
    finally:
        session.close()


def test_no_arguments_is_no_block(tmp_path: Path) -> None:
    tree = Tree(tmp_path)
    write_skill(tree.project, "style")
    session = new_session(tmp_path)
    try:
        got = invoke(Skills(discover(tree.roots()), session), "style", "  ", call_id="u")
        assert isinstance(got, Invoked) and got.arguments is None
        assert got.render() == got.body
    finally:
        session.close()
