"""`locate` declares `UNAVAILABLE`: the scout returns it when
it cannot search, and a failure outside a tool's declared `failures` is a harness bug.
Zeus's description says what to do on it."""

from __future__ import annotations

from pathlib import Path

import chickadee.adapters.frontier as frontier
from chickadee.tools.spec import SURFACE

TEMPLATE = Path(frontier.__file__).parent / "templates" / "tools" / "locate.md"


def test_locate_declares_unavailable() -> None:
    (locate,) = [s for s in SURFACE if s.name == "locate"]
    assert "UNAVAILABLE" in locate.failures
    assert {"UNRESOLVED", "TIMED_OUT"} <= locate.failures


def test_the_template_names_it_and_what_zeus_does_instead() -> None:
    text = TEMPLATE.read_text(encoding="utf-8")
    assert "UNAVAILABLE" in text
    for fallback in ("symbol", "grep", "glob", "grep_in"):
        assert fallback in text, fallback
