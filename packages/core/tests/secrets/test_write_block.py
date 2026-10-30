"""SE-3 — the applier rejects a patch targeting a deny-listed path.

Mirrors D20's test-file lockdown, with the same exception rule: allowed iff the path is
itself the declared target **and** it matches a `.example`/`.template` negation.
"""

from __future__ import annotations

import pytest

from chickadee.secrets.denylist import Denylist, write_denied


def blocked(path: str) -> bool:
    """Thin wrapper over the **production** rule, so this file tests the shipped code.

    The first version reimplemented SE-3 here, which meant the test proved its own helper
    worked and said nothing about what the applier would do.
    """
    return write_denied(path) is not None


@pytest.mark.parametrize(
    "path", [".env", ".env.production", "certs/prod.pem", ".aws/credentials"]
)
def test_writing_a_secret_path_is_refused(path: str) -> None:
    assert blocked(path)


@pytest.mark.parametrize("path", [".env", "certs/prod.pem"])
def test_a_secret_path_is_refused_regardless_of_any_declaration(path: str) -> None:
    """SE-3 does not consult the packet's declarations at all, and that is the point.

    Same shape as D20: a packet cannot argue its way in through the scope declaration the
    gate exists to constrain. Here it is structural rather than checked — there is no
    argument to make, because the rule has no input that could carry one. Applier-level
    coverage lives with #92.
    """
    assert blocked(path)


def test_writing_the_example_is_not_refused_by_se3() -> None:
    """Writing `.env.example` is legitimate work — it documents the names.

    SE-3 permits it because a negation makes it readable. Whether it is a *declared target*
    is gate 2's question, and the spec's "iff" comes from composing the two rather than from
    either one restating the other.
    """
    assert not blocked(".env.example")


def test_the_example_is_readable_and_therefore_writable() -> None:
    """The negation is the whole reason the exception is safe: a template holds names."""
    assert Denylist().readable(".env.example")
    assert not Denylist().readable(".env")


def test_se3_is_silent_about_ordinary_files() -> None:
    """It answers one question — is this path secret-bearing — and nothing else."""
    assert not blocked("README.md")
    assert not blocked("src/app.py")


def test_the_rule_lives_in_one_place() -> None:
    """A rule expressed twice disagrees with itself the first time either copy is edited.

    This asserts the gate has something to call rather than something to re-derive.
    """
    from chickadee.secrets import denylist

    assert callable(denylist.write_denied)
    denial = denylist.write_denied(".env")
    assert denial is not None
    assert "denied: secret-bearing path" in denial.reason
