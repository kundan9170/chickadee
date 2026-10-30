"""Isolation wrap: `SandboxBackend.wrap(Isolation)`, not `ExecRequest`.

This package must not import `chickadee.executor`.
"""

from chickadee.isolation.base import SandboxBackend, select
from chickadee.isolation.types import Isolation, SandboxLevel

__all__ = ["Isolation", "SandboxBackend", "SandboxLevel", "select"]
