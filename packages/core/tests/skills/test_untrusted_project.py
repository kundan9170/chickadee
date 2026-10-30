"""SK-8, IJ-4: under `repo_trust = third_party` a project skill's load writes `untrusted_read`
and its body is framed as untrusted; without the person's acceptance it does not load. User
skills are never gated."""

from __future__ import annotations

from pathlib import Path

from pydantic import TypeAdapter
from skills_helpers import Tree, new_session, write_skill

from chickadee.context.results import InvalidArgument
from chickadee.skills.discover import discover
from chickadee.skills.index import render_index
from chickadee.skills.load import Skills
from chickadee.trust.events import UNTRUSTED_READ, UntrustedRead


def test_an_accepted_project_skill_loads_framed_untrusted_and_journals_the_crossing(
    tmp_path: Path,
) -> None:
    tree = Tree(tmp_path)
    path = write_skill(tree.project, "deploy", body="Deploy with make ship.\n")
    session = new_session(tmp_path)
    try:
        skills = Skills(
            discover(tree.roots()), session, repo_trust="third_party", accepted=True
        )
        got = skills.load("deploy", "k1")
        assert isinstance(got, str) and got.splitlines()[1].startswith("UNTRUSTED:")
        (event,) = [e for e, _ in session.state.lines() if e.kind == UNTRUSTED_READ]
        line = TypeAdapter(UntrustedRead).validate_json(session.blob(event.payload_sha))
        assert line.path == str(path) and line.bytes == len("Deploy with make ship.\n")
    finally:
        session.close()


def test_an_unaccepted_project_skill_does_not_load(tmp_path: Path) -> None:
    tree = Tree(tmp_path)
    write_skill(tree.project, "deploy")
    session = new_session(tmp_path)
    try:
        got = Skills(discover(tree.roots()), session, repo_trust="third_party").load(
            "deploy", "k"
        )
        assert isinstance(got, InvalidArgument) and "not accepted" in got.why
        assert session.state.last_seq == 0
    finally:
        session.close()


def test_a_user_skill_is_never_gated(tmp_path: Path) -> None:
    tree = Tree(tmp_path)
    write_skill(tree.user, "mine")
    session = new_session(tmp_path)
    try:
        got = Skills(discover(tree.roots()), session, repo_trust="third_party").load(
            "mine", "k"
        )
        assert isinstance(got, str) and "UNTRUSTED" not in got
        assert not [e for e, _ in session.state.lines() if e.kind == UNTRUSTED_READ]
    finally:
        session.close()


def test_the_index_shows_a_project_rows_origin_under_third_party(tmp_path: Path) -> None:
    tree = Tree(tmp_path)
    write_skill(tree.project, "deploy", "how we deploy")
    write_skill(tree.user, "mine", "my habits")
    index = render_index(discover(tree.roots()), repo_trust="third_party")
    assert (
        "- deploy · project · [untrusted · from this repository: deploy.md] how we deploy"
        in index
    )
    assert "- mine · user · my habits" in index
