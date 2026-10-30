"""The diff a stop shows beside a part's README (`workspace.tree_diff`): the
repository's own diff program never runs in it."""

from __future__ import annotations

import stat
from pathlib import Path

from repos import git, init_repo

from chickadee.workspace import tree_diff


def test_the_repositorys_external_diff_never_runs(tmp_path: Path) -> None:
    root = init_repo(tmp_path / "repo", {"design/README.md": "# A\n"})
    before = git(root, "rev-parse", "HEAD^{tree}")
    (root / "design" / "README.md").write_text("# A\n\nAnd B.\n")
    git(root, "add", "-A")
    git(root, "commit", "-q", "-m", "b")
    after = git(root, "rev-parse", "HEAD^{tree}")
    # What a command could plant: a program git runs for every diff it shows.
    ran = tmp_path / "ran"
    program = tmp_path / "differ"
    program.write_text(f"#!/bin/sh\ntouch {ran}\n")
    program.chmod(program.stat().st_mode | stat.S_IXUSR)
    git(root, "config", "diff.external", str(program))

    shown = tree_diff(root, before, after, "the design approved before")
    assert shown is not None and "+And B." in shown
    assert not ran.exists(), "the repository's diff program ran"
