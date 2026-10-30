"""Built-in slash-command names are reserved. A skill named for one is refused at
index time, naming the file and the built-in; `RESERVED_NAMES` is `12-cli.md` the table, read
from the document so the two cannot drift."""

from __future__ import annotations

from pathlib import Path

import pytest
from skills_helpers import Tree, write_skill

from chickadee.errors import ConfigError
from chickadee.skills.discover import discover


@pytest.mark.parametrize("scope", ["project", "user", "external"])
def test_a_skill_named_status_refuses_at_index_time_naming_both(
    tmp_path: Path, scope: str
) -> None:
    tree = Tree(tmp_path)
    root = {"project": tree.project, "user": tree.user, "external": tree.external()}[scope]
    path = write_skill(root, "status", bundle=scope == "external")
    with pytest.raises(ConfigError) as refused:
        discover(tree.roots())
    message = str(refused.value)
    assert str(path) in message and "`/status`" in message
