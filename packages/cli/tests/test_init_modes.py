"""A file `init` replaces keeps its mode; a new one is 0644 (D149 DU-3).

`atomic_replace` wrote through `mkstemp`, which creates 0600, and `os.replace` carries the
temp file's mode onto the target. So every re-init made `state.json` owner-only, and adding
the exclude line did the same to `.git/info/exclude`, a file that belongs to git and to the
person. The applier's writer already kept the mode; `T-3.91` makes the two agree.
"""

from __future__ import annotations

import os
import pathlib
import stat

from chickadee.init import detect, init, publish


def _mode(path: pathlib.Path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


def test_a_new_state_file_is_0644(repo: pathlib.Path) -> None:
    init(repo)
    assert _mode(publish.state_path(repo)) == 0o644


def test_a_reinit_keeps_the_state_files_mode(repo: pathlib.Path) -> None:
    init(repo)
    state = publish.state_path(repo)
    os.chmod(state, 0o640)
    init(repo, force=True)
    assert _mode(state) == 0o640


def test_adding_the_exclude_line_keeps_the_exclude_files_mode(repo: pathlib.Path) -> None:
    exclude = detect.exclude_path(repo)
    exclude.parent.mkdir(parents=True, exist_ok=True)
    exclude.write_text("# git ls-files --others --exclude-from=.git/info/exclude\n")
    os.chmod(exclude, 0o644)
    init(repo)
    assert publish.GIT_EXCLUDE_LINE in exclude.read_text().splitlines()
    assert _mode(exclude) == 0o644
