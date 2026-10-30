"""Every tracked path can be checked out on Windows and macOS.

Nine journal fixtures were named `operations/apply:p1:1:1/manifest.json`, and a clone on
Windows failed on them. This reads the repository's own file list, so a path no Windows
checkout can create fails here, wherever it is added.
"""

from __future__ import annotations

import re
import shutil
from collections import defaultdict
from pathlib import Path

import pytest
from repos import git

REPO_ROOT = Path(__file__).resolve().parents[3]

ILLEGAL = re.compile(r'[<>:"|?*\\\x00-\x1f]')
"""Characters Windows refuses in a file name."""

RESERVED = re.compile(r"(?i)^(con|prn|aux|nul|com[1-9]|lpt[1-9])(\..*)?$")
"""Device names Windows refuses as a file name, with or without an extension."""


def _tracked() -> list[str]:
    if shutil.which("git") is None or not (REPO_ROOT / ".git").exists():
        pytest.skip("not a git checkout")
    return [p for p in git(REPO_ROOT, "ls-files", "-z").split("\0") if p]


def problems(paths: list[str]) -> list[str]:
    """Every path a Windows or macOS checkout cannot create, with the reason."""
    found: list[str] = []
    folded: defaultdict[str, list[str]] = defaultdict(list)
    for path in paths:
        folded[path.casefold()].append(path)
        for part in path.split("/"):
            if ILLEGAL.search(part):
                found.append(f"{path}: {part!r} holds a character Windows refuses")
            elif RESERVED.match(part):
                found.append(f"{path}: {part!r} is a Windows device name")
            elif part.endswith((".", " ")):
                found.append(f"{path}: {part!r} ends in a dot or a space")
    found += [f"{v}: differ only in case" for v in folded.values() if len(v) > 1]
    return found


def test_every_tracked_path_is_portable() -> None:
    assert problems(_tracked()) == []


@pytest.mark.parametrize(
    "path",
    [
        "fixtures/operations/apply:p1:1:1/manifest.json",
        "a/b?.txt",
        "docs/CON",
        "docs/aux.md",
        "trailing./x",
        "space /x",
    ],
)
def test_a_path_windows_refuses_is_reported(path: str) -> None:
    assert len(problems([path])) == 1


def test_two_paths_differing_only_in_case_are_reported() -> None:
    assert problems(["docs/Readme.md", "docs/README.md"]) != []


def test_an_encoded_operation_directory_is_portable() -> None:
    assert problems(["fixtures/operations/apply%3Ap1%3A1%3A1/manifest.json"]) == []
