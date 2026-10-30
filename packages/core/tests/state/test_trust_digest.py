"""Executable config is declared by the project and trusted by the person."""

from __future__ import annotations

import pathlib

import pytest

from chickadee.config import SecretsPolicy
from chickadee.state.trust import (
    SHRINKS_THE_DENY_LIST,
    TRUST_FILE,
    acceptance_record,
    executable_subset,
    is_active,
)


def accept(config, repo_local: pathlib.Path) -> str:
    """What the CLI does with `acceptance_record`: show, ask, then write the body.

    Local to the tests because `state/` deliberately performs no disk writes — the package
    decides, the surface persists.
    """
    digest, body = acceptance_record(config, repo_local)
    repo_local.mkdir(parents=True, exist_ok=True)
    (repo_local / TRUST_FILE).write_text(body, encoding="utf-8")
    return digest


HOOKED = {
    "profiles": "python",
    "hooks": [{"event": "pre_action", "command": "./evil.sh", "on_error": "allow"}],
}


def test_a_config_with_no_executable_parts_is_active(tmp_path: pathlib.Path) -> None:
    """Nothing to accept, so nothing to ask about. The worst a wrong non-executable value
    does is make the harness refuse to work."""
    assert is_active({"profiles": "python"}, tmp_path) is True


def test_a_cloned_hook_is_inert_until_accepted(tmp_path: pathlib.Path) -> None:
    """The whole point: cloning a repository must not be enough to run its author's code."""
    assert is_active(HOOKED, tmp_path) is False
    accept(HOOKED, tmp_path)
    assert is_active(HOOKED, tmp_path) is True


def test_editing_the_command_returns_it_to_inert(tmp_path: pathlib.Path) -> None:
    """Accepting a hook once never accepts its replacement."""
    accept(HOOKED, tmp_path)
    edited = {**HOOKED, "hooks": [{**HOOKED["hooks"][0], "command": "./other.sh"}]}
    assert is_active(edited, tmp_path) is False


def test_changing_only_non_executable_config_keeps_acceptance(tmp_path: pathlib.Path) -> None:
    """The digest covers the executable subset, not the whole file — otherwise every
    unrelated edit would re-ask, and re-asking about nothing is how people stop reading."""
    accept(HOOKED, tmp_path)
    assert is_active({**HOOKED, "profiles": "rust"}, tmp_path) is True


def test_acceptance_does_not_travel_with_the_repository(tmp_path: pathlib.Path) -> None:
    """A repository that could ship its own acceptance would be asserting its own
    trustworthiness — so the record lives in repo-local state, which a clone does not carry.

    Modelled as two clones of the same project: accepting in one leaves the other inert.
    """
    clone_a = tmp_path / "a" / ".git" / "chickadee"
    clone_b = tmp_path / "b" / ".git" / "chickadee"
    accept(HOOKED, clone_a)

    assert (clone_a / TRUST_FILE).is_file()
    assert is_active(HOOKED, clone_a) is True
    assert is_active(HOOKED, clone_b) is False


def test_the_record_is_readable_text(tmp_path: pathlib.Path) -> None:
    """Plain TOML on purpose: a person must be able to read what they accepted."""
    digest = accept(HOOKED, tmp_path)
    body = (tmp_path / TRUST_FILE).read_text(encoding="utf-8")
    assert digest in body
    assert "accepted" in body


def test_subset_covers_the_three_named_surfaces() -> None:
    """Identified by name, never guessed — a subset inferred by what *seems* executable
    would silently stop covering a field someone adds later."""
    config = {
        "hooks": [{"event": "e", "command": "c"}],
        "mcp": {"servers": [{"name": "jira", "command": "x"}]},
        "executor": {"env": {"passthrough": ["PATH"]}},
        "profiles": "python",
    }
    subset = executable_subset(config).subset
    assert set(subset) == {"hooks", "mcp.servers", "executor.env"}
    assert "profiles" not in subset


# --- D118 item 3: a removal from the deny list is executable config ---


def _denylist(config, repo_local: pathlib.Path):
    """What a run resolves: the `[secrets]` block, applied as far as this clone accepted."""
    policy = SecretsPolicy.model_validate(config.get("secrets", {}))
    return policy.denylist(accepted=is_active(config, repo_local))


REMOVES_ENV = {"profiles": "python", "secrets": {"deny_remove": [".env"]}}


def test_adding_a_removal_moves_the_digest() -> None:
    """A digest that did not move would be a credential decision propagating silently —
    the failure §6 exists to prevent."""
    before = executable_subset({"profiles": "python"})
    after = executable_subset(REMOVES_ENV)
    assert before.is_empty and not after.is_empty
    assert after.subset == {"secrets.deny_remove": [".env"]}
    widened = executable_subset(
        {"profiles": "python", "secrets": {"deny_remove": [".env", ".npmrc"]}}
    )
    assert widened.digest != after.digest, "removing one more entry is a new decision"


def test_a_clone_that_has_not_accepted_resolves_the_list_without_the_removal(
    tmp_path: pathlib.Path,
) -> None:
    """Cloning a repository is not consent to expose your `.env` to it; accepting
    the digest once is."""
    clone = tmp_path / "clone" / ".git" / "chickadee"
    assert not _denylist(REMOVES_ENV, clone).readable(".env")
    accept(REMOVES_ENV, clone)
    assert _denylist(REMOVES_ENV, clone).readable(".env")
    other = tmp_path / "other" / ".git" / "chickadee"
    assert not _denylist(REMOVES_ENV, other).readable(".env"), "acceptance does not travel"


def test_allow_secret_read_in_a_repository_is_inert_until_accepted(
    tmp_path: pathlib.Path,
) -> None:
    config = {"secrets": {"allow_secret_read": True}}
    assert executable_subset(config).subset == {"secrets.allow_secret_read": True}
    assert not _denylist(config, tmp_path).allow_secret_read
    accept(config, tmp_path)
    assert _denylist(config, tmp_path).allow_secret_read


def test_additions_and_spelled_out_defaults_are_not_executable_config(
    tmp_path: pathlib.Path,
) -> None:
    """An addition only tightens, so it is live on a fresh clone — an inert one would leave
    the clone looser than its project. `false` and `[]` shrink nothing, so a config that
    spells the default out loud is not asked about."""
    config = {
        "secrets": {"deny_add": ["*.kdbx"], "deny_remove": [], "allow_secret_read": False}
    }
    assert executable_subset(config).is_empty
    assert is_active(config, tmp_path)
    assert not _denylist(config, tmp_path).readable("vault.kdbx")
    assert set(SHRINKS_THE_DENY_LIST) == {"secrets.deny_remove", "secrets.allow_secret_read"}


def test_a_value_is_judged_the_way_the_config_reads_it(tmp_path: pathlib.Path) -> None:
    """Judged by the config, never by truthiness. A `false` shrinks nothing, so a line that
    spells the default keeps the acceptance. A value the config cannot read counts, the
    direction that asks the owner: since `T-3.44` reads booleans strictly, a quoted `"false"`
    is one, refused at load and asked about here, where it once read as false."""
    accept(HOOKED, tmp_path)
    assert is_active(HOOKED, tmp_path)

    off = {**HOOKED, "secrets": {"allow_secret_read": False}}
    assert "secrets.allow_secret_read" not in executable_subset(off).subset
    assert is_active(off, tmp_path), "a line that changes nothing keeps the acceptance"

    quoted = {**HOOKED, "secrets": {"allow_secret_read": "false"}}
    with pytest.raises(ValueError, match="allow_secret_read"):
        SecretsPolicy.model_validate(quoted["secrets"])
    assert "secrets.allow_secret_read" in executable_subset(quoted).subset

    unreadable = {**HOOKED, "secrets": {"allow_secret_read": "maybe"}}
    assert "secrets.allow_secret_read" in executable_subset(unreadable).subset

    turned_on = {**HOOKED, "secrets": {"allow_secret_read": "true"}}
    assert not is_active(turned_on, tmp_path), "a real shrink still needs acceptance"
