"""SK-11: a compaction that evicts everything evictable leaves a loaded skill BYTE-IDENTICAL.

The regression this guards is the one the market shipped: after a
compaction an agent "completely loses awareness of Skills that were actively being used". A
pinned body is lifted out of the cut region and re-emitted intact.
"""

from __future__ import annotations

from pathlib import Path

from skills_helpers import Tree, context_text, filler, load, must_fit, new_session, write_skill

from chickadee.skills.discover import discover
from chickadee.skills.load import Skills

BODY = "# Migrations\n\nAlways use online DDL on the events table,\nexcept below 10k rows.\n"


def test_a_must_fit_cut_leaves_the_loaded_body_byte_identical(tmp_path: Path) -> None:
    tree = Tree(tmp_path)
    write_skill(tree.project, "migrations", body=BODY)
    session = new_session(tmp_path)
    try:
        filler(session)
        loaded = load(session, Skills(discover(tree.roots()), session), "migrations", "k1")
        assert BODY.rstrip() in loaded
        filler(session)
        done = must_fit(session)
        assert done.cut is not None
        assert len(done.cut.lifted) == 1, "the load's tool line is lifted"
        after = context_text(session)
        assert loaded in after, "re-emitted byte for byte"
    finally:
        session.close()


def test_a_body_not_loaded_is_not_lifted(tmp_path: Path) -> None:
    tree = Tree(tmp_path)
    write_skill(tree.project, "migrations", body=BODY)
    session = new_session(tmp_path)
    try:
        filler(session, 10)
        done = must_fit(session)
        assert done.cut is not None and done.cut.lifted == ()
        assert BODY.strip() not in context_text(session)
    finally:
        session.close()
