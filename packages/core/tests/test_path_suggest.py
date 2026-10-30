"""`paths.suggest`: the file a path that does not resolve most likely meant,
and the read rule's files never among them."""

from __future__ import annotations

from pathlib import Path

import pytest

from chickadee.graph.ignore import listed
from chickadee.paths import suggest, suggest_under
from chickadee.secrets.denylist import Denylist

FILES = ("config.py", "core/engine.py", "core/x/util.py", "tests/test_engine.py", "a/u.py")


def test_the_longest_suffix_that_is_a_file() -> None:
    assert suggest("/workspaces/game/config.py", FILES) == "config.py"
    assert suggest("/workspaces/game/core/engine.py", FILES) == "core/engine.py"
    assert suggest("/srv/core/x/util.py", FILES) == "core/x/util.py"


def test_else_the_one_file_of_that_name() -> None:
    assert suggest("src/engine.py", FILES) == "core/engine.py"


def test_else_none() -> None:
    assert suggest("/workspaces/game/main.py", FILES) is None
    assert suggest("", FILES) is None
    assert suggest("/", FILES) is None


def test_a_name_two_files_share_is_none() -> None:
    assert suggest("lib/util.py", (*FILES, "b/util.py")) is None


@pytest.mark.parametrize(
    "sent",
    [
        "'/workspaces/game/config.py'",
        '"config.py"',
        "`config.py`",
        "file:///workspaces/game/config.py",
        "C:\\workspaces\\game\\config.py",
        "./config.py",
        "//workspaces//game/./config.py",
    ],
)
def test_the_path_is_read_loosely_first(sent: str) -> None:
    assert suggest(sent, FILES) == "config.py"


def test_a_new_file_s_directory() -> None:
    assert suggest_under("/workspaces/game/core/new.py", FILES) == "core/new.py"
    assert suggest_under("/workspaces/game/nowhere/new.py", FILES) is None
    assert suggest_under("new.py", FILES) is None


def _repo(root: Path) -> Path:
    for rel, text in {
        "config.py": "x = 1\n",
        ".env": "TOKEN=abc\n",
        "build/config.py": "generated\n",
        "secrets/.env": "TOKEN=abc\n",
        ".gitignore": "build/\nlocal.py\n",
        "sub/local.py": "x = 2\n",
    }.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(text, encoding="utf-8")
    return root


def test_a_deny_listed_or_ignored_file_is_never_suggested(tmp_path: Path) -> None:
    files = listed(_repo(tmp_path), Denylist())
    assert "config.py" in files
    assert not {".env", "secrets/.env", "build/config.py", "sub/local.py"} & set(files)
    assert suggest("/workspaces/r/.env", files) is None
    assert suggest("/workspaces/r/local.py", files) is None
    assert suggest("/workspaces/r/build/config.py", files) == "config.py"
