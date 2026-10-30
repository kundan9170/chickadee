"""SK-12, SK-13: a bundled script is never loaded. `load_skill` reads `SKILL.md` alone; a
`scripts/` program is the Executor's to run, by the path the body names, and its source never
enters context through a load. (Its acceptance in the trust digest is a follow-up the task's
Log names.)"""

from __future__ import annotations

from pathlib import Path

from skills_helpers import Tree, new_session, write_skill

from chickadee.skills.discover import discover
from chickadee.skills.load import Skills

SCRIPT = "import sys\nprint('VALIDATOR-SOURCE-LINE')\nsys.exit(0)\n"
REFERENCE = "REFERENCE-ONLY: the long table of column types.\n"


def test_a_load_carries_the_body_and_none_of_the_bundle(tmp_path: Path) -> None:
    tree = Tree(tmp_path)
    path = write_skill(
        tree.project,
        "migrations",
        body="Run `scripts/check.py` before proposing DDL; see REFERENCE.md for types.\n",
        bundle=True,
    )
    (path.parent / "scripts").mkdir()
    (path.parent / "scripts" / "check.py").write_text(SCRIPT, encoding="utf-8")
    (path.parent / "REFERENCE.md").write_text(REFERENCE, encoding="utf-8")
    session = new_session(tmp_path)
    try:
        got = Skills(discover(tree.roots()), session).load("migrations", "k1")
        assert isinstance(got, str)
        assert "scripts/check.py" in got, "the body names it"
        assert "VALIDATOR-SOURCE-LINE" not in got and "REFERENCE-ONLY" not in got
    finally:
        session.close()


def test_a_bundle_is_one_skill_not_one_per_file(tmp_path: Path) -> None:
    tree = Tree(tmp_path)
    path = write_skill(tree.project, "migrations", bundle=True)
    (path.parent / "REFERENCE.md").write_text(REFERENCE, encoding="utf-8")
    assert [s.name for s in discover(tree.roots()).skills] == ["migrations"]
