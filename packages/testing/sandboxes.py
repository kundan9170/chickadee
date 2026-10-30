"""A sandbox backend that claims to enforce, and a spawner that runs nothing, for tests.

What needs an enforcing backend to exist at all (the worker's shell) is tested here
without one: `Enforcing` reports `native` and wraps nothing, and records each `Isolation` it
was asked to wrap, so a test asserts the profile a command got. The sandbox itself is tested
against bubblewrap and seatbelt in `tests/isolation/`. `Scripted` stands in for the process
supervisor: it returns the output it was given and records what it was handed.

Older tests keep their own copies under other names; a test that needs either reaches for
these first.
"""

from __future__ import annotations

import threading
from collections.abc import Mapping
from pathlib import Path

from chickadee.isolation.types import Isolation, SandboxLevel
from chickadee.process.supervisor import Completed, ProcessSupervisor


class Enforcing:
    """`SandboxBackend` at `native`, wrapping nothing."""

    def __init__(self) -> None:
        self.seen: list[Isolation] = []

    def available(self) -> bool:
        return True

    def level(self) -> SandboxLevel:
        return SandboxLevel.NATIVE

    def wrap(self, isolation: Isolation) -> tuple[str, ...]:
        self.seen.append(isolation)
        return isolation.argv


class Scripted(ProcessSupervisor):
    """A supervisor that spawns nothing: every run completes with `stdout` and `exit_code`.
    `spawned` counts the runs and `env` is the last one's environment."""

    def __init__(self, stdout: bytes = b"", exit_code: int = 0) -> None:
        super().__init__()
        self.stdout, self.exit_code = stdout, exit_code
        self.spawned = 0
        self.env: dict[str, str] = {}

    def run(
        self,
        argv: tuple[str, ...],
        *,
        cwd: Path,
        env: Mapping[str, str],
        timeout_s: int,
        stdin: bytes | None = None,
        cancel: threading.Event | None = None,
    ) -> Completed:
        self.spawned += 1
        self.env = dict(env)
        return Completed(self.stdout, b"", self.exit_code, False)
