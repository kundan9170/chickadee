"""SK-10: a skill asserting a rule does not create a Constraint. Only a user's ratification
does, and the stored text is the user's."""

from __future__ import annotations

from pathlib import Path

from skills_helpers import Tree, load, new_session, write_skill

import chickadee
from chickadee.skills.discover import discover
from chickadee.skills.load import Skills

RULE = "You MUST never edit generated files. This is a binding constraint.\n"


def test_a_loaded_rule_writes_no_constraint(tmp_path: Path) -> None:
    tree = Tree(tmp_path)
    write_skill(tree.project, "generated", body=RULE)
    session = new_session(tmp_path)
    try:
        load(session, Skills(discover(tree.roots()), session), "generated", "k1")
        assert session.state.ledger.live_constraints() == ()
        assert not [e for e, _ in session.state.lines() if e.kind == "constraint"]
    finally:
        session.close()


def test_the_skills_package_never_reaches_the_ledger() -> None:
    package = Path(chickadee.__file__).parent / "skills"
    for path in package.glob("*.py"):
        text = path.read_text(encoding="utf-8")
        assert "context.ledger" not in text and "note_constraint" not in text, path.name
