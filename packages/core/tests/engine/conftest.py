"""A minimal valid config, so each test can break exactly one thing."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest
from engines import raw_config

from chickadee.engine import PacketEngine


@pytest.fixture
def raw() -> dict[str, Any]:
    """`engines.raw_config()`: one model per rung, each named unlike its rung."""
    return raw_config()


@pytest.fixture
def engines() -> Iterator[list[PacketEngine]]:
    """Every engine a test opens, closed when it ends."""
    opened: list[PacketEngine] = []
    yield opened
    for engine in opened:
        engine.close()
