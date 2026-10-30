"""Fixtures for the graph tests. Real files on disk and a real SQLite database — the graph
is an integration between tree-sitter, the filesystem and SQL, and a mocked version of any
of the three tests the mock."""

from __future__ import annotations

from pathlib import Path

import pytest
from repos import init_repo

from chickadee.graph.indexer import Indexer
from chickadee.graph.queries import Graph

APP = '''\
"""App."""
from helpers import helper

MAX_SIZE = 10
counter: int = 0


class Base:
    """Base."""

    def setup(self):
        return None


class Widget(Base):
    """A widget."""

    def render(self, x: int) -> str:
        """Render it."""
        return helper(x)


def build_widget(name: str) -> Widget:
    """Build one."""
    return Widget()
'''

HELPERS = '''\
"""Helpers."""


def helper(x: int) -> str:
    """Help."""
    return str(x)


def getUserName(user):
    """camelCase on purpose, for GR-3."""
    return user


def refresh_session(session):
    """snake_case on purpose, for GR-3."""
    return session
'''

TEST_APP = """\
from app import build_widget


def test_build_widget():
    assert build_widget("a") is not None
"""


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    root.mkdir()
    (root / "app.py").write_text(APP, encoding="utf-8")
    (root / "helpers.py").write_text(HELPERS, encoding="utf-8")
    (root / "tests").mkdir()
    (root / "tests" / "test_app.py").write_text(TEST_APP, encoding="utf-8")
    init_repo(root, commit=None)
    return root


@pytest.fixture
def indexed(repo: Path) -> Graph:
    ix = Indexer(repo)
    ix.full_index()
    return Graph(repo, ix.conn)
