"""Journal payloads for the three things this package observes.

**Names, paths, kinds and fingerprints — never values.** A journal is durable and replayable,
so a secret written into one is a secret with a long life and many copies.
"""

from __future__ import annotations

from dataclasses import dataclass

from chickadee.secrets.denylist import Denial
from chickadee.secrets.redact import Redaction
from chickadee.secrets.suspect import Finding


@dataclass(frozen=True)
class SecretDenied:
    path: str
    pattern: str
    surface: str
    """Which read surface asked — scout, packer, planner, reader, graph."""

    @staticmethod
    def of(denial: Denial, surface: str) -> SecretDenied:
        return SecretDenied(denial.path, denial.pattern, surface)


@dataclass(frozen=True)
class SecretSuspected:
    path: str
    line: int
    kind: str
    detail: str

    @staticmethod
    def of(finding: Finding) -> SecretSuspected:
        return SecretSuspected(finding.path, finding.line, finding.kind, finding.detail)


@dataclass(frozen=True)
class RedactionApplied:
    kind: str
    fingerprint: str
    count: int
    target: str
    """Which blob the redaction happened on the way into."""

    @staticmethod
    def of(redaction: Redaction, target: str) -> RedactionApplied:
        return RedactionApplied(redaction.kind, redaction.fingerprint, redaction.count, target)
