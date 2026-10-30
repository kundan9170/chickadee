"""An advisory pass that reports and never cuts.

Runs over an assembled context bundle looking for values that *look* like hardcoded
credentials, and returns findings. **It mutates nothing.** The packer calls it; the packer
still packs.

That restraint is the decision, not an oversight. BL-47 records why denial was rejected: the
packer cannot tell a hardcoded secret from a test fixture, a documentation sample, or a
variable someone named `secret_sauce`, so blanking content would be a correctness failure
*caused by* a security feature. Warning produces the data that would settle whether denial
is ever safe; blanking produces a broken run and no data.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

from chickadee.edits.lines import split_lines
from chickadee.secrets.redact import DETECTORS

MIN_ENTROPY = 3.5
"""Shannon bits per character. Below this a string is words, not a key."""

MIN_LENGTH = 20

ASSIGNMENT = re.compile(
    rf"(?i)\b(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*[=:]\s*[\"'](?P<value>[^\"'\n]{{{MIN_LENGTH},}})[\"']"
)

BENIGN_WORDS = frozenset(
    {
        "sauce",
        "santa",
        "section",
        "sector",
        "secretary",
        "sample",
        "example",
        "fixture",
        "dummy",
        "placeholder",
        "lorem",
        "test",
    }
)
"""Words that make an identifier benign despite a trigger substring.

Matched against the identifier's **subwords**, not with `\b`: `_` is a word character, so
`\bexample\b` never fires inside `example_token` and the canonical fixture constant was
reported anyway. `secret_sauce` is the other direction — a trigger substring in a name that
means nothing of the kind."""


def _subwords(name: str) -> set[str]:
    parts = re.split(r"[^A-Za-z0-9]+", name)
    out: set[str] = set()
    for part in parts:
        out.update(
            w.lower() for w in re.findall(r"[A-Z]+(?![a-z])|[A-Z][a-z]*|[a-z]+|\d+", part)
        )
    return out


def _benign(name: str) -> bool:
    return bool(_subwords(name) & BENIGN_WORDS)


@dataclass(frozen=True)
class Finding:
    path: str
    line: int
    kind: str
    detail: str


def shannon(value: str) -> float:
    if not value:
        return 0.0
    counts = {c: value.count(c) for c in set(value)}
    total = len(value)
    return -sum((n / total) * math.log2(n / total) for n in counts.values())


def scan_text(path: str, text: str) -> tuple[Finding, ...]:
    """Findings for one file's content. Path and line included so a report can point."""
    findings: list[Finding] = []

    # Known formats are matched against the **whole text**, then mapped back to a line.
    # Scanning line by line meant the `private_key` detector — which needs both PEM
    # delimiters — could never fire, so a real multiline private key was reported as
    # nothing at all.
    # One finding per line, and the **first** detector wins. `DETECTORS` runs specific
    # formats before the generic `assigned_secret`, so `API_KEY = "ghp_..."` is reported as
    # a GitHub token rather than twice — once precisely and once vaguely.
    flagged_lines: set[int] = set()
    for detector in DETECTORS:
        for match in detector.pattern.finditer(text):
            line_no = text.count("\n", 0, match.start()) + 1
            if line_no in flagged_lines:
                continue
            findings.append(
                Finding(path, line_no, detector.kind, "matches a known credential format")
            )
            flagged_lines.add(line_no)

    for number, line in enumerate(split_lines(text), start=1):
        if number in flagged_lines:
            continue
        match = ASSIGNMENT.search(line)
        if match is None:
            continue
        name, value = match.group("name"), match.group("value")
        if _benign(name):
            continue
        if _looks_like_prose(value) or shannon(value) < MIN_ENTROPY:
            continue
        findings.append(
            Finding(path, number, "high_entropy", f"{name} assigned a high-entropy literal")
        )
    return tuple(sorted(findings, key=lambda f: f.line))


def _looks_like_prose(value: str) -> bool:
    """A sentence, a path, or a URL is long and varied without being a key."""
    return " " in value.strip() or value.startswith(("/", "./", "http://", "https://"))


def scan(files: dict[str, str]) -> tuple[Finding, ...]:
    """Advisory sweep over an assembled bundle. Returns findings; changes nothing."""
    out: list[Finding] = []
    for path, text in sorted(files.items()):
        out.extend(scan_text(path, text))
    return tuple(out)
