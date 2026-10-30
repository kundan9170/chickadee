"""Pins before `T-3.157` item 8: what the loader answers today for each key
`approval.toml` may hold, read through `parse_level`.

Item 8 moved some of these on purpose: `always_ask`, `never_ask` and `[approval.facts]` are
refused at load naming D196, and the class rules of
`[approval.commands]` and the paths of `deny` reach the shell and the edit port. These pins
were the "before" that change is measured against; the moved answers are pinned as they are
now, and what stays (`deny`, `grants`, an unknown key, a gate-shaped key) stays.
"""

from __future__ import annotations

import pytest

from chickadee.approval.prefs import PreferenceError, parse_level


def _level(body: dict[str, object]) -> object:
    return parse_level({"approval": body}, "user")


def test_deny_loads_as_path_globs() -> None:
    v = parse_level({"approval": {"deny": ["secrets/**"]}}, "repo")
    assert v.deny == ("secrets/**",)


# Moved by item 8: each of these loaded before (`always_ask` and `never_ask` as path
# globs, a fact row as `ask` or `allow`); each set the work tier, which D196 §12.3 removed, and
# is refused at load naming it, so no rule a file states is silently not kept.


@pytest.mark.parametrize(
    "body",
    [
        {"always_ask": ["migrations/**"]},
        {"never_ask": ["docs/**"]},
        {"facts": {"test_file": "allow"}},
        {"facts": {"test_file": "deny"}},
        {"facts": {"destructive_git": "allow"}},
    ],
    ids=["always_ask", "never_ask", "facts", "facts-deny", "facts-hard-allow"],
)
def test_the_work_tier_keys_are_refused(body: dict[str, object]) -> None:
    with pytest.raises(PreferenceError, match="D196"):
        _level(body)


def test_a_command_class_loads_and_a_hard_class_may_not_be_allowed() -> None:
    v = parse_level({"approval": {"commands": {"lockfile_install": "allow"}}}, "user")
    assert dict(v.commands) == {"lockfile_install": "allow"}
    ask = parse_level({"approval": {"commands": {"tests": "ask"}}}, "user")
    assert dict(ask.commands) == {"tests": "ask"}
    with pytest.raises(PreferenceError):
        _level({"commands": {"push": "allow"}})


def test_grants_load_as_not_preferences() -> None:
    v = parse_level({"approval": {"grants": []}}, "repo")
    assert (v.deny, dict(v.commands)) == ((), {})


def test_an_unknown_key_and_a_gate_shaped_key_are_refused() -> None:
    with pytest.raises(PreferenceError, match="unknown key"):
        _level({"colour": "blue"})
    with pytest.raises(PreferenceError, match="disables a gate"):
        _level({"skip_checks": True})
    with pytest.raises(PreferenceError, match="only \\[approval\\]"):
        parse_level({"other": {}}, "user")
