"""SE-9 — one server gets only its declared names, and never a value in a model surface.

The rule the design states twice: *credential values are injected only at the authorized
boundary; models and journals receive names or opaque refs, never values.*
"""

from __future__ import annotations

import pytest

from chickadee.secrets.broker import Broker, UndeclaredCredential
from chickadee.secrets.events import SecretDenied
from chickadee.secrets.redact import redact

VAULT = {
    "GITHUB_TOKEN": "ghp_1234567890abcdefghij",
    "DATABASE_URL": "postgres://u:hunter2@db/app",
    "SLACK_TOKEN": "xoxb-1234567890-abcdefg",
}


def test_a_server_receives_only_its_declared_names() -> None:
    """Declaring one credential does not hand over the drawer."""
    injected = Broker(VAULT).inject(("GITHUB_TOKEN",), approved=("GITHUB_TOKEN",))
    assert set(injected) == {"GITHUB_TOKEN"}
    assert "DATABASE_URL" not in injected
    assert "SLACK_TOKEN" not in injected


def test_two_servers_do_not_share_a_scope() -> None:
    broker = Broker(VAULT)
    github = broker.inject(("GITHUB_TOKEN",), approved=("GITHUB_TOKEN",))
    database = broker.inject(("DATABASE_URL",), approved=("DATABASE_URL",))
    assert set(github) & set(database) == set()


def test_a_name_not_approved_raises_rather_than_being_dropped() -> None:
    """Silently omitting it hands the call a missing variable and lets it fail somewhere
    confusing, while the authorization mismatch — the thing that actually went wrong — goes
    unreported."""
    with pytest.raises(UndeclaredCredential, match="not approved"):
        Broker(VAULT).inject(("GITHUB_TOKEN", "SLACK_TOKEN"), approved=("GITHUB_TOKEN",))


def test_the_approval_gate_renders_names_not_values() -> None:
    """You approve *which credential names are exposed to which command*, so the gate must
    be able to say that without printing the secret."""
    exposure = Broker(VAULT).exposure(("GITHUB_TOKEN", "DATABASE_URL"))
    assert exposure.summary == "exposes: GITHUB_TOKEN, DATABASE_URL"
    for value in VAULT.values():
        assert value not in exposure.summary


def test_a_credential_bearing_exposure_is_flagged_for_second_confirmation() -> None:
    """D40 class. Computed here rather than eyeballed at the gate."""
    assert Broker(VAULT).exposure(("GITHUB_TOKEN",)).credential_bearing
    assert not Broker(VAULT).exposure(("LANG",)).credential_bearing


def test_no_value_reaches_a_model_or_journal_surface() -> None:
    """The end of SE-9: names and refs travel, values do not."""
    exposure = Broker(VAULT).exposure(("GITHUB_TOKEN",))
    event = SecretDenied(path=".env", pattern=".env", surface="packer")
    prompt = f"Use the credential named {exposure.names[0]} when calling the server.\n"

    surfaces = repr(exposure) + repr(event) + prompt
    for value in VAULT.values():
        assert value not in surfaces


def test_a_leaked_value_is_still_redacted_before_persistence() -> None:
    """Belt and braces: if a value reaches a blob anyway, layer 3 catches the known shape."""
    stored, found = redact(f"server responded with {VAULT['GITHUB_TOKEN']}".encode())
    assert VAULT["GITHUB_TOKEN"].encode() not in stored
    assert found
