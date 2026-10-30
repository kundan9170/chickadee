"""SK-2: no body text appears in the assembled prefix before a load. The index holds names,
scopes and descriptions, and nothing else."""

from __future__ import annotations

from pathlib import Path

from skills_helpers import Tree, write_skill

from chickadee.executor.types import SandboxLevel
from chickadee.roles import role
from chickadee.skills.discover import discover
from chickadee.skills.index import render_index
from chickadee.tools.spec import SURFACE
from chickadee.work import DEFAULT_DIR
from chickadee.zeus.prefix import Env, assemble

BODY = "SECRET-SAUCE: the backfill runs in batches of 500 with a 2s sleep.\n"


def test_the_prefix_carries_the_description_and_never_the_body(tmp_path: Path) -> None:
    tree = Tree(tmp_path)
    write_skill(tree.project, "migrations", "how migrations are written here", body=BODY)
    write_skill(tree.user, "style", "house style", body=BODY, bundle=True)
    index = render_index(discover(tree.roots()))
    prefix = assemble(
        role("zeus"),
        SURFACE,
        index,
        Env(
            workspace=str(tree.repo),
            platform="linux",
            sandbox=SandboxLevel.NONE,
            reads_masked=False,
            model="m",
            repo_trust="first_party",
        ),
        work_dir=DEFAULT_DIR,
    )
    assert "how migrations are written here" in prefix.text
    assert "SECRET-SAUCE" not in prefix.text
    assert "load_skill" in prefix.text
