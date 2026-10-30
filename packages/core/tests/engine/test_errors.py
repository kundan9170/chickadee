"""Every public error is distinct, catchable as one base, and `HarnessBug` is loud."""

from __future__ import annotations

import inspect

import pytest

from chickadee import errors
from chickadee.errors import ChickadeeError, HarnessBug

PUBLIC = [
    obj
    for _, obj in inspect.getmembers(errors, inspect.isclass)
    if issubclass(obj, ChickadeeError) and obj is not ChickadeeError
]

# the table is the contract a host writes `except` clauses against.
EXPECTED = {
    "ConfigError",
    "EngineBusy",
    "RepoLeaseBusy",
    "RepoNotIndexed",
    "PlannerError",
    "InfraError",
    "HarnessBug",
    "SessionError",
    "RunNotStarted",
    "HandoffError",
    "AttachmentRefused",
}


def test_the_hierarchy_is_exactly_the_documented_table() -> None:
    """In both directions: a class added here without a row is undocumented API, and a row
    without a class is a promise the host cannot catch."""
    assert {c.__name__ for c in PUBLIC} == EXPECTED


@pytest.mark.parametrize("cls", PUBLIC, ids=lambda c: c.__name__)
def test_every_public_error_shares_one_catchable_base(cls: type) -> None:
    assert issubclass(cls, ChickadeeError)
    with pytest.raises(ChickadeeError):
        raise cls("boom")


@pytest.mark.parametrize("cls", PUBLIC, ids=lambda c: c.__name__)
def test_no_public_error_is_a_subclass_of_another(cls: type) -> None:
    """Flat on purpose. If `InfraError` caught `HarnessBug`, a host handling infrastructure
    failure would silently swallow our own invariant violations — the one class §5 says
    must always be reported."""
    others = [c for c in PUBLIC if c is not cls]
    assert not any(issubclass(cls, o) for o in others), cls.__name__


def test_harness_bug_is_not_caught_by_the_others() -> None:
    for cls in PUBLIC:
        if cls is HarnessBug:
            continue
        with pytest.raises(HarnessBug):
            try:
                raise HarnessBug("invariant violated")
            except cls:  # must not match
                pytest.fail(f"{cls.__name__} swallowed HarnessBug")
