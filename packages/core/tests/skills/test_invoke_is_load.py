"""SK-14: invoking by name reaches context through the SAME loader as a model selection: same
pin, same size refusal, same trust gate."""

from __future__ import annotations

from pathlib import Path

from skills_helpers import Tree, new_session, write_skill

from chickadee.context.results import InvalidArgument
from chickadee.skills.discover import discover
from chickadee.skills.invoke import Invoked, invoke
from chickadee.skills.load import SKILL_BODY_MAX, Skills
from chickadee.skills.pin import pinned


def test_an_invocation_pins_through_the_loader(tmp_path: Path) -> None:
    tree = Tree(tmp_path)
    write_skill(tree.project, "migrations", body="# M\n\nbackfill in batches\n")
    session = new_session(tmp_path)
    try:
        skills = Skills(discover(tree.roots()), session)
        got = invoke(skills, "migrations", "events", call_id="u1")
        assert isinstance(got, Invoked) and "backfill in batches" in got.body
        assert [(p.name, p.call_id) for p in pinned(session)] == [("migrations", "u1")]
    finally:
        session.close()


def test_an_invocation_meets_the_same_refusals(tmp_path: Path) -> None:
    tree = Tree(tmp_path)
    write_skill(tree.project, "huge", body="x" * int(SKILL_BODY_MAX * 3.5 + 100))
    write_skill(tree.project, "theirs")
    session = new_session(tmp_path)
    try:
        refused = invoke(Skills(discover(tree.roots()), session), "huge", "", call_id="u1")
        assert isinstance(refused, InvalidArgument) and "nothing was loaded" in refused.why
        gated = Skills(discover(tree.roots()), session, repo_trust="third_party")
        unaccepted = invoke(gated, "theirs", "", call_id="u2")
        assert isinstance(unaccepted, InvalidArgument) and "not accepted" in unaccepted.why
        assert pinned(session) == ()
    finally:
        session.close()
