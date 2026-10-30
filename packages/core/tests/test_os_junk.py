"""`paths.os_junk`: the files an operating system's file browser writes on its own,
by name at any depth and in any case, and nothing else."""

from __future__ import annotations

import pytest

from chickadee.paths import os_junk


@pytest.mark.parametrize(
    "path",
    [
        ".DS_Store",
        "docs/chickadee/.DS_Store",
        "a/b/Thumbs.db",
        "thumbs.db",
        "desktop.ini",
        "x/Desktop.ini",
        "ehthumbs.db",
    ],
)
def test_an_os_file_is_junk_at_any_depth_and_case(path: str) -> None:
    assert os_junk(path)


@pytest.mark.parametrize(
    "path",
    ["README.md", ".DS_Store.md", "docs/DS_Store", "my.DS_Store", ".DS_Store/x.md", "thumbs"],
)
def test_a_file_that_only_looks_like_one_is_not(path: str) -> None:
    assert not os_junk(path)
