"""SK-11: a cut at any pressure re-emits a pinned body BYTE-IDENTICAL, never collapsed,
projected, summarised or truncated."""

from __future__ import annotations

from pathlib import Path

import pytest
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

from chickadee.skills.discover import discover
from chickadee.skills.load import Skills

BODY = "".join(f"rule {n}: keep the {n}th invariant, with its reason.\n" for n in range(60))


@pytest.mark.parametrize("cuts", [1, 3])
def test_every_cut_re_emits_the_body_intact(tmp_path: Path, cuts: int) -> None:
    tree = Tree(tmp_path)
    write_skill(tree.project, "rules", body=BODY)
    session = new_session(tmp_path)
    driver = compaction()
    try:
        filler(session)
        loaded = load(session, Skills(discover(tree.roots()), session), "rules", "k1")
        for _ in range(cuts):
            filler(session)
            assert must_fit(session, driver).cut is not None
        text = context_text(session)
        assert text.count(BODY.rstrip()) == 1
        assert loaded in text
    finally:
        session.close()
