"""Layer 2 — a spawned process cannot see an undeclared variable."""

from __future__ import annotations

from chickadee.secrets.env import PASSTHROUGH, looks_like_credential, scrub

PARENT = {
    "PATH": "/usr/bin",
    "HOME": "/home/u",
    "AWS_SECRET_ACCESS_KEY": "abc123",
    "DATABASE_URL": "postgres://u:p@h/db",
    "GITHUB_TOKEN": "ghp_x",
    "MY_APP_SETTING": "on",
}


def test_an_undeclared_variable_is_absent() -> None:
    env = scrub(PARENT)
    assert "AWS_SECRET_ACCESS_KEY" not in env
    assert "DATABASE_URL" not in env
    assert "GITHUB_TOKEN" not in env


def test_a_declared_variable_reaches_the_process() -> None:
    env = scrub(PARENT, declared=("DATABASE_URL",))
    assert env["DATABASE_URL"] == "postgres://u:p@h/db"
    assert "GITHUB_TOKEN" not in env, "declaring one does not declare the rest"


def test_the_passthrough_default_is_what_a_process_needs_to_run() -> None:
    env = scrub(PARENT)
    assert set(env) == {"PATH", "HOME"}
    assert all(name in PASSTHROUGH for name in env)


def test_it_is_an_allowlist_not_a_denylist() -> None:
    """A denylist of credential-shaped names passes anything unusually named, and the
    failure is silent — the process starts fine and the value is simply there."""
    env = scrub({"CONN": "postgres://u:p@h/db", "DSN": "secret", "PATH": "/usr/bin"})
    assert set(env) == {"PATH"}


def test_a_declared_but_absent_variable_is_simply_missing() -> None:
    env = scrub(PARENT, declared=("NOT_SET_ANYWHERE",))
    assert "NOT_SET_ANYWHERE" not in env


def test_credential_shape_is_advisory_only() -> None:
    """It makes the approval gate louder; it never decides what is passed."""
    assert looks_like_credential("GITHUB_TOKEN")
    assert looks_like_credential("db_password")
    assert not looks_like_credential("LANG")
    env = scrub(PARENT, declared=("GITHUB_TOKEN",))
    assert env["GITHUB_TOKEN"] == "ghp_x", "advisory naming does not block a declared name"
