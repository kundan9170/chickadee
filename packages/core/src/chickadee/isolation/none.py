"""The passthrough that reports `level=none` loudly.

Invariant 6 applied to isolation: an un-sandboxed run is a fact about the result, not a
detail to hide. This backend exists so that fact has somewhere to be recorded — the
alternative is silently running unwrapped and letting every downstream reader assume
otherwise.
"""

from __future__ import annotations

from chickadee.isolation.types import Isolation, SandboxLevel


def reads_unmasked(level: SandboxLevel, isolation: Isolation) -> bool:
    """EX-2, one more field: a run at `level=none` with a non-empty
    `deny_read` could not mask anything, and the `exec` event and the run header must
    say so rather than let the reader assume the mask held. Computed, not stored: a
    fact about this wrap, exactly like the level itself."""
    return level is SandboxLevel.NONE and bool(isolation.deny_read)


class Passthrough:
    def available(self) -> bool:
        return True

    def level(self) -> SandboxLevel:
        return SandboxLevel.NONE

    def wrap(self, isolation: Isolation) -> tuple[str, ...]:
        return isolation.argv
