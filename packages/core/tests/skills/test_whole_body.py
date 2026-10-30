"""A load returns the entire document. No partial body is reachable through `load_skill`,
at any size under the cap, and the load does not go through the read path."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
from skills_helpers import Tree, new_session, write_skill

from chickadee.skills import load as load_module
from chickadee.skills.discover import discover
from chickadee.skills.load import SKILL_BODY_MAX, Skills


@pytest.mark.parametrize("lines", [1, 200, 600])
def test_the_whole_body_arrives(tmp_path: Path, lines: int) -> None:
    body = "".join(
        f"line {n}: the rule and its exception, stated together.\n" for n in range(lines)
    )
    tree = Tree(tmp_path)
    write_skill(tree.project, "rules", body=body)
    session = new_session(tmp_path)
    try:
        got = Skills(discover(tree.roots()), session).load("rules", "k1")
        assert isinstance(got, str)
        assert body.rstrip() in got
        assert got.count("line ") == lines
    finally:
        session.close()
    assert len(body) / 3.5 < SKILL_BODY_MAX


def test_the_loader_never_imports_the_read_path() -> None:
    """§5.1 step 3: not the read path of `28-context.md` §3, and not routed through it."""
    tree = ast.parse(Path(load_module.__file__).read_text(encoding="utf-8"))
    imported = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }
    assert not {
        m
        for m in imported
        if m.startswith(("chickadee.context.reader", "chickadee.context.read"))
    }
