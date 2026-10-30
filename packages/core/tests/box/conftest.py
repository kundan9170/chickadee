"""Box fixtures: a session's box over a fresh committed repository."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from box_helpers import BoxRig, make_repo


@pytest.fixture
def rig(tmp_path: Path) -> Iterator[BoxRig]:
    r = BoxRig(make_repo(tmp_path))
    yield r
    r.close()
