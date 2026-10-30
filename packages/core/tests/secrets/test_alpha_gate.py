"""The trusted prototype's allowance does not open the alpha gates.

§3.5 states it plainly: the prototype's first-party credential allowance "does not authorize
third-party MCP, external alpha users, or untrusted repos. Those modes fail closed until the
alpha security gate passes."

These assert it **fails closed**, because a rule that only lives in a document is one the
first integration walks past.
"""

from __future__ import annotations

import pytest

from chickadee.secrets.broker import AlphaGateClosed, Broker

VAULT = {"GITHUB_TOKEN": "ghp_1234567890abcdefghij"}
ONE = ("GITHUB_TOKEN",)


def test_the_trusted_first_party_prototype_is_allowed() -> None:
    """The allowance the prototype does have, so the gate is not simply 'no'."""
    broker = Broker(VAULT, repo_trust="first_party")
    assert broker.inject(ONE, approved=ONE, server_trust="first_party") == VAULT


def test_a_third_party_server_fails_closed() -> None:
    broker = Broker(VAULT, repo_trust="first_party")
    with pytest.raises(AlphaGateClosed, match="third-party"):
        broker.inject(ONE, approved=ONE, server_trust="third_party")


def test_an_untrusted_repository_fails_closed() -> None:
    broker = Broker(VAULT, repo_trust="third_party")
    with pytest.raises(AlphaGateClosed, match="untrusted repositories"):
        broker.inject(ONE, approved=ONE, server_trust="first_party")


def test_the_repository_gate_is_checked_even_for_a_first_party_server() -> None:
    """Both conditions are independent; passing one does not carry the other."""
    broker = Broker(VAULT, repo_trust="third_party")
    with pytest.raises(AlphaGateClosed):
        broker.inject(ONE, approved=ONE, server_trust="first_party")


def test_passing_the_gate_opens_both() -> None:
    """The gate is a gate, not a wall: once the alpha security work lands it opens."""
    broker = Broker(VAULT, repo_trust="third_party", alpha_gate_passed=True)
    assert broker.inject(ONE, approved=ONE, server_trust="third_party") == VAULT


def test_the_default_is_closed() -> None:
    """A Broker built without an opinion does not quietly grant the alpha modes."""
    broker = Broker(VAULT)
    assert broker.repo_trust == "first_party"
    assert not broker.alpha_gate_passed
    with pytest.raises(AlphaGateClosed):
        broker.inject(ONE, approved=ONE, server_trust="third_party")


def test_the_refusal_names_which_gate_closed() -> None:
    """Two different gates with two different fixes — a shared message would hide which."""
    repo_closed = Broker(VAULT, repo_trust="third_party")
    server_closed = Broker(VAULT, repo_trust="first_party")
    with pytest.raises(AlphaGateClosed) as repo_error:
        repo_closed.inject(ONE, approved=ONE, server_trust="first_party")
    with pytest.raises(AlphaGateClosed) as server_error:
        server_closed.inject(ONE, approved=ONE, server_trust="third_party")
    assert str(repo_error.value) != str(server_error.value)
