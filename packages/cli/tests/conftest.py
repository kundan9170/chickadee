"""A real git repository per test — init resolves identity through git plumbing.

Fixtures only. A plain helper here would have to be imported as `from conftest import...`,
and the test directories are not packages, so that name resolves to whichever `conftest`
pytest loaded first across the whole suite.
"""

from __future__ import annotations

import pathlib

import pytest
from repos import init_repo


@pytest.fixture
def repo(tmp_path: pathlib.Path) -> pathlib.Path:
    return init_repo(
        tmp_path / "repo", {"app.py": "def main() -> None:\n    pass\n"}, commit=None
    )
