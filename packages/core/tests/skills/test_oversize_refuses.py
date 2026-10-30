"""SK-9: a skill over `SKILL_BODY_MAX` refuses by name and loads NOTHING, rather than
truncating to fit."""

from __future__ import annotations

from pathlib import Path

from skills_helpers import Tree, new_session, write_skill

from chickadee.context.results import InvalidArgument
from chickadee.skills.discover import discover
from chickadee.skills.load import SKILL_BODY_MAX, Skills
from chickadee.skills.pin import pinned


def test_an_oversize_body_is_refused_naming_the_file_and_nothing_is_recorded(
    tmp_path: Path,
) -> None:
    tree = Tree(tmp_path)
    path = write_skill(tree.project, "huge", body="x" * int(SKILL_BODY_MAX * 3.5 + 100))
    session = new_session(tmp_path)
    try:
        before = session.state.last_seq
        got = Skills(discover(tree.roots()), session).load("huge", "k1")
        assert isinstance(got, InvalidArgument)
        assert str(path) in got.why and str(SKILL_BODY_MAX) in got.why
        assert "nothing was loaded" in got.why
        assert session.state.last_seq == before and pinned(session) == ()
    finally:
        session.close()
