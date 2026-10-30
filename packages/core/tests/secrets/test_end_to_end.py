"""The whole point, checked once end to end.

A fixture repo with a `.env` and a test that prints `os.environ`: no secret value may reach
any blob, prompt or journal payload.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from chickadee.secrets.denylist import Denylist
from chickadee.secrets.env import scrub
from chickadee.secrets.events import RedactionApplied, SecretDenied, SecretSuspected
from chickadee.secrets.redact import redact
from chickadee.secrets.suspect import scan

SECRET = "ghp_1234567890abcdefghij"


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    root.mkdir()
    (root / ".env").write_text(f"GITHUB_TOKEN={SECRET}\n", encoding="utf-8")
    (root / ".env.example").write_text("GITHUB_TOKEN=\n", encoding="utf-8")
    (root / "app.py").write_text("def go():\n    return 1\n", encoding="utf-8")
    return root


def test_no_secret_value_reaches_any_persisted_surface(repo: Path) -> None:
    denylist = Denylist()
    persisted: list[bytes] = []

    # Layer 1 — the packer never reads it, so the value never enters context.
    context: dict[str, str] = {}
    denied: list[SecretDenied] = []
    for path in sorted(p.relative_to(repo).as_posix() for p in repo.iterdir()):
        hit = denylist.match(path)
        if hit is not None:
            denied.append(SecretDenied.of(hit, surface="packer"))
            continue
        context[path] = (repo / path).read_text(encoding="utf-8")

    assert any(d.path == ".env" for d in denied)
    assert ".env.example" in context, "the template is readable and carries only names"
    assert not any(SECRET in text for text in context.values())

    # Layer 2 — a spawned process cannot inherit it.
    env = scrub({**os.environ, "GITHUB_TOKEN": SECRET})
    assert "GITHUB_TOKEN" not in env

    # Layer 3 — anything that leaks anyway is redacted before the write.
    leaked = f"Traceback: connecting with GITHUB_TOKEN={SECRET}\n".encode()
    stored, redactions = redact(leaked)
    persisted.append(stored)
    events = [RedactionApplied.of(r, target="exec_stdout") for r in redactions]

    # The advisory pass reports, and changes nothing.
    suspected = [SecretSuspected.of(f) for f in scan(context)]

    haystack = b"".join(persisted) + repr(denied + events + suspected).encode()
    assert SECRET.encode() not in haystack, "a secret reached a persisted surface"
    assert b"REDACTED:github_pat" in haystack, "and the fact of it was still recorded"


def test_the_denial_is_recorded_without_the_value(repo: Path) -> None:
    """A journal is durable and replayable, so a secret written into one has a long life
    and many copies. Names, paths and fingerprints travel; values do not."""
    hit = Denylist().match(".env")
    assert hit is not None
    event = SecretDenied.of(hit, surface="scout")
    assert SECRET not in repr(event)
    assert event.path == ".env"
