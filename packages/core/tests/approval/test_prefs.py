"""Two-level preference precedence.

Command verdicts compose across the levels, the highest floor winning; a denial anywhere
denies; an unrecognised verdict escalates; a preference cannot disable a gate, and the work
tier's keys are refused naming D196 — the loader refuses both.

The tier the router computed from these (`approval/compute.py`, `path_floor`, the claims and run
points) retired with the work tier; what this file pins is what the
loader answers.
"""

from __future__ import annotations

import pathlib

import pytest

from chickadee.approval.commands import CommandClass, shipped
from chickadee.approval.prefs import (
    PreferenceError,
    command_denied,
    denied,
    load_level,
    load_preferences,
    merge,
    parse_level,
)
from chickadee.approval.tier import Preferences, Tier
from chickadee.state.roots import Roots


def prefs_from(repo: str = "", user: str = "") -> Preferences:
    import tomllib

    levels = []
    if repo:
        levels.append(parse_level(tomllib.loads(repo), "repo", source="repo"))
    if user:
        levels.append(parse_level(tomllib.loads(user), "user", source="user"))
    return merge(levels)


def command_denied_for(prefs: Preferences, *argv: str) -> CommandClass | None:
    classified = shipped().classify(argv)
    assert classified is not None, f"{argv} is a classified command"
    return command_denied(prefs, classified)


# --- the work tier's keys are refused -------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        '[approval]\nalways_ask = ["tests/**"]\n',
        '[approval]\nnever_ask = ["tests/**"]\n',
        '[approval.facts]\ntest_file = "allow"\n',
        '[approval.facts]\ntest_file = "deny"\n',
        '[approval.facts]\ndestructive_git = "allow"\n',
    ],
    ids=["always_ask", "never_ask", "facts", "facts-deny", "facts-hard-allow"],
)
def test_a_work_tier_key_refuses_the_load(text: str) -> None:
    """Each set the work tier, which D196 §12.3 removed: refused with that reason and what to
    use instead, never loaded and silently not kept."""
    with pytest.raises(PreferenceError, match=r"D196.*approval\.deny"):
        prefs_from(user=text)


# --- two levels compose; caution accumulates -------------------------------------------------


def test_the_order_of_levels_does_not_matter() -> None:
    """`max()` is commutative, so no level wins — the most cautious answer does."""
    a = prefs_from(
        repo='[approval.commands]\nlockfile_install = "allow"\n',
        user='[approval.commands]\nlockfile_install = "ask"\n',
    )
    b = prefs_from(
        repo='[approval.commands]\nlockfile_install = "ask"\n',
        user='[approval.commands]\nlockfile_install = "allow"\n',
    )
    assert (
        a.command_overrides["lockfile_install"]
        is b.command_overrides["lockfile_install"]
        is Tier.T1
    )


# --- [approval.commands] reaches the argv's row ---


def test_a_command_verdict_lowers_the_lowerable_class_at_dispatch() -> None:
    """`lockfile_install = "allow"` is the user's judgment on the one T1 class; the pre-spawn
    hit for `uv sync` goes to T0 and the lowering is journaled under `command:`."""
    prefs = prefs_from(user='[approval.commands]\nlockfile_install = "allow"\n')
    assert prefs.command_overrides["lockfile_install"] is Tier.T0
    assert prefs.source["command:lockfile_install"] == "user"


def test_a_command_verdict_raises_a_t0_class_even_though_it_fires_no_hit() -> None:
    """`read_only = "ask"`: `ls` fires no row, and the user still asked to be asked."""
    prefs = prefs_from(repo='[approval.commands]\nread_only = "ask"\n')
    assert prefs.command_overrides["read_only"] is Tier.T1
    assert prefs.source["command:read_only"] == "repo"
    assert prefs_from().command_overrides == {}


def test_command_verdicts_compose_across_levels_like_row_verdicts() -> None:
    prefs = prefs_from(
        repo='[approval.commands]\nlockfile_install = "ask"\n',
        user='[approval.commands]\nlockfile_install = "allow"\n',
    )
    assert prefs.command_overrides["lockfile_install"] is Tier.T1
    assert prefs.source["command:lockfile_install"] == "repo"


def test_a_command_class_set_to_deny_is_refused_before_the_tier() -> None:
    """`"deny"` was composed as an `ask` floor and nothing refused the command."""
    prefs = prefs_from(repo='[approval.commands]\npush = "deny"\n')
    assert prefs.denied_commands == {"push"}
    assert command_denied_for(prefs, "git", "push", "origin", "main") is CommandClass.PUSH
    # Any command the line runs counts: a denied push behind `&&` is still a push.
    line = ("sh", "-c", "ls && git push origin main")
    assert command_denied_for(prefs, *line) is CommandClass.PUSH
    assert command_denied_for(prefs, "git", "status") is None


def test_a_class_denied_at_either_level_is_denied() -> None:
    prefs = prefs_from(
        repo='[approval.commands]\npush = "ask"\n',
        user='[approval.commands]\npush = "deny"\n',
    )
    assert prefs.denied_commands == {"push"}


def test_deny_is_path_globs_read_as_gitignore_reads_them() -> None:
    """`deny` matches through `paths.glob_match`: `*` stays in one segment, a
    glob with no `/` is a name at any depth."""
    prefs = prefs_from(user='[approval]\ndeny = ["app/*.py", "*.lock"]\n')
    assert denied(prefs, "app/x.py") == "app/*.py"
    assert denied(prefs, "app/sub/x.py") is None
    assert denied(prefs, "pkg/uv.lock") == "*.lock"


# --- unrecognised escalates; unknown keys refuse ---


def test_an_unrecognised_verdict_escalates_to_ask() -> None:
    prefs = prefs_from(user='[approval.commands]\nlockfile_install = "whatever"\n')
    assert prefs.command_overrides["lockfile_install"] is Tier.T1


def test_an_unknown_rule_name_refuses_the_load() -> None:
    with pytest.raises(PreferenceError, match="unknown key"):
        prefs_from(user='[approval.commands]\nlockfile_instal = "allow"\n')
    with pytest.raises(PreferenceError, match="unknown key"):
        prefs_from(user='[approval]\nnever_asks = ["x"]\n')


def test_a_hard_class_cannot_be_allowed() -> None:
    with pytest.raises(PreferenceError, match="hard floor"):
        prefs_from(user='[approval.commands]\npush = "allow"\n')
    # Asking for more is always allowed.
    assert (
        prefs_from(user='[approval.commands]\npush = "ask"\n').command_overrides["push"]
        is Tier.T1
    )


# --- a preference cannot disable a gate ------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        '[approval]\nskip_gates = ["A1"]\n',
        "[approval]\ndisable_lockdown = true\n",
        '[approval.gates]\na1 = "off"\n',
        '[approval.commands]\ntest_lockdown = "allow"\n',
        '[approval]\nverification = "never"\n',
    ],
)
def test_a_preference_that_would_disable_a_gate_refuses_the_load(text: str) -> None:
    with pytest.raises(PreferenceError, match="never whether anything is recorded or checked"):
        prefs_from(user=text)


# --- the files -------------------------------------------------------------------------------


def test_load_reads_both_locations_and_a_missing_file_is_an_empty_level(
    tmp_path: pathlib.Path,
) -> None:
    project = tmp_path / "repo"
    (project / ".chickadee").mkdir(parents=True)
    (project / ".chickadee" / "approval.toml").write_text('[approval]\ndeny = ["infra/**"]\n')
    user = tmp_path / "xdg" / "chickadee"
    user.mkdir(parents=True)
    (user / "approval.toml").write_text('[approval]\ndeny = ["docs/**"]\n')
    # `project` as `state.roots` resolves it: the repository's `.chickadee/`, not its root.
    roots = Roots(
        project=project / ".chickadee",
        repo_local=project / ".git" / "chickadee",
        checkout_local=project / ".git" / "chickadee",
        user=user,
    )
    prefs = load_preferences(roots)
    assert prefs.deny == ("infra/**", "docs/**")
    assert (prefs.source["deny:infra/**"], prefs.source["deny:docs/**"]) == ("repo", "user")

    (user / "approval.toml").unlink()
    assert load_preferences(roots).deny == ("infra/**",)


def test_malformed_toml_refuses_with_the_path(tmp_path: pathlib.Path) -> None:
    bad = tmp_path / "approval.toml"
    bad.write_text("[approval\n")
    with pytest.raises(PreferenceError, match="malformed TOML"):
        load_level(bad, "user")


def test_grants_are_accepted_and_ignored_here() -> None:
    """D91's `[[approval.grants]]` live in the same file and are read elsewhere."""
    prefs = prefs_from(repo='[[approval.grants]]\neffect = "expose_path"\npath = ".env"\n')
    assert prefs == Preferences()
