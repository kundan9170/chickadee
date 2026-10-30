"""Layer 2 — never inherit.

A spawned process does not receive `os.environ`. It receives an allowlist, plus whatever an
action packet **declared** — and declaring is visible at the approval gate, which is the
point: you approve *which credential names are exposed to which command*, not a blanket
grant.

This is the Codex `shell_environment_policy` pattern — inject the specific value at runtime
rather than handing over the whole environment.
"""

from __future__ import annotations

from collections.abc import Mapping

PASSTHROUGH: tuple[str, ...] = ("PATH", "HOME", "LANG", "LC_ALL", "TZ", "TERM")
"""Built-in default. Everything else is absent unless an action names it."""

CREDENTIAL_MARKERS = (
    "TOKEN",
    "SECRET",
    "PASSWORD",
    "APIKEY",
    "API_KEY",
    "CREDENTIAL",
    "AUTH",
    "KEY",
)


def scrub(
    parent: Mapping[str, str],
    declared: tuple[str, ...] = (),
    passthrough: tuple[str, ...] = PASSTHROUGH,
) -> dict[str, str]:
    """The environment a spawned process actually gets.

    An **allowlist**, not a denylist. A denylist of credential-shaped names would pass
    anything unusually named — `DB_DSN`, `CONN` — and the failure mode is silent: the
    process starts fine and the value is simply there. Starting from nothing means a missing
    variable is a loud, immediate error instead.
    """
    keep = {k: v for k, v in parent.items() if k in passthrough}
    for name in declared:
        if name in parent:
            keep[name] = parent[name]
    return keep


def looks_like_credential(name: str) -> bool:
    """Advisory only — used to render "exposes: DATABASE_URL" more loudly at the approval
    gate. It never decides what is passed; `declared` does."""
    upper = name.upper()
    return any(marker in upper for marker in CREDENTIAL_MARKERS)
