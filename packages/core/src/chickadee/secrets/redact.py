"""Layer 3 — never persist.

Every byte on its way into a blob passes here: prompts, completions, `exec` output, digests,
findings, transcripts.

**A backstop, and honestly labelled as one.** Layers 1 and 2 are prevention; this is
detection, so it is incomplete by construction — it catches known shapes and misses novel
ones. Treating it as the protection would be the mistake.

**Applied at the boundary, once, before the write.** Never as a display-time filter: a
display filter leaves the value on disk, which is the failure this exists to prevent.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

PLACEHOLDER = "[REDACTED:{kind}:{fingerprint}]"


@dataclass(frozen=True)
class Detector:
    kind: str
    pattern: re.Pattern[str]
    group: int = 0
    """Which capture carries the secret. Some patterns must match surrounding context —
    `password =...` — while replacing only the value."""


DETECTORS: tuple[Detector, ...] = (
    Detector("aws_access_key", re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b")),
    Detector("github_pat", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{16,}\b")),
    Detector("openai_key", re.compile(r"\bsk-(?!ant-)[A-Za-z0-9_-]{16,}\b")),
    Detector("anthropic_key", re.compile(r"\bsk-ant-[A-Za-z0-9_-]{16,}\b")),
    Detector("slack_token", re.compile(r"\bxox[bpar]-[A-Za-z0-9-]{10,}\b")),
    Detector("google_key", re.compile(r"\bAIza[A-Za-z0-9_-]{35}\b")),
    Detector(
        "jwt", re.compile(r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\b")
    ),
    Detector(
        "private_key",
        re.compile(
            r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----", re.S
        ),
    ),
    Detector(
        "url_credentials",
        re.compile(r"(?<=://)([^/\s:@]+:[^/\s:@]+)(?=@)"),
        group=1,
    ),
    Detector(
        "assigned_secret",
        re.compile(
            r"(?i)\b(?:password|secret|token|api[_-]?key)\s*[=:]\s*[\"']?([^\s\"'`,;]{12,})"
        ),
        group=1,
    ),
)


@dataclass(frozen=True)
class Redaction:
    kind: str
    fingerprint: str
    count: int


def fingerprint(value: str) -> str:
    """Eight hex characters of the value's digest.

    Identity without the value: the *same* secret produces the *same* placeholder
    everywhere, so a reader can say "that is the token from turn 3" while holding nothing
    they could use. A random id per occurrence would destroy that; the raw value would
    destroy the point.
    """
    return hashlib.sha256(value.encode("utf-8", "replace")).hexdigest()[:8]


def redact_text(text: str) -> tuple[str, tuple[Redaction, ...]]:
    counts: dict[tuple[str, str], int] = {}

    for detector in DETECTORS:

        def replace(
            match: re.Match[str], kind: str = detector.kind, group: int = detector.group
        ) -> str:
            value = match.group(group)
            token = PLACEHOLDER.format(kind=kind, fingerprint=fingerprint(value))
            counts[(kind, fingerprint(value))] = counts.get((kind, fingerprint(value)), 0) + 1
            whole = match.group(0)
            return whole.replace(value, token) if group else token

        text = detector.pattern.sub(replace, text)

    found = tuple(Redaction(kind, fp, n) for (kind, fp), n in sorted(counts.items()))
    return text, found


def redact(data: bytes) -> tuple[bytes, tuple[Redaction, ...]]:
    """The blob-write path. Bytes in, bytes out, so a caller cannot forget to re-encode.

    `surrogateescape`, not `replace`. This sits on the way into every blob, and `replace`
    is lossy in both directions: `b"\xff"` came back as `b"\xef\xbf\xbd"` even with no
    secret present, so a redactor that found nothing still corrupted the bytes it was
    handed. Surrogate escaping round-trips arbitrary bytes exactly.
    """
    text, found = redact_text(data.decode("utf-8", "surrogateescape"))
    return text.encode("utf-8", "surrogateescape"), found
