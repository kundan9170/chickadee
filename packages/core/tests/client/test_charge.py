"""The one charge of a failed call (`client/charge.py`)."""

from __future__ import annotations

import ast
from pathlib import Path

import chickadee
from chickadee.client.charge import charge_unanswered, unanswered_usage
from chickadee.client.errors import ContextOverflow, RequestCancelled, TransportExhausted
from chickadee.schema.adapter import Usage

SRC = Path(chickadee.__file__).parent


def test_each_billed_attempt_is_its_input_and_what_it_streamed() -> None:
    failed = TransportExhausted("malformed", 3, billed_attempts=(0, 7, 3))
    assert unanswered_usage(100, failed) == Usage(
        prompt_tokens=300, completion_tokens=10, estimated=True
    )


def test_nothing_billed_is_nothing_charged() -> None:
    charged: list[Usage] = []
    for failed in (
        TransportExhausted("HTTP 503", 4),
        ContextOverflow("too long", 400, 1),
        RequestCancelled(0),
    ):
        assert unanswered_usage(100, failed) is None
        assert charge_unanswered(lambda u: charged.append(u) or 1.0, _never, failed) == 0.0
    assert charged == []


def _never() -> int:
    """The request's estimate, which a failure no attempt of was billed never asks for: an
    outage's failed calls would each walk the whole conversation for nothing."""
    raise AssertionError("the estimate was asked for a call nothing billed")


def test_the_charge_is_what_the_ledger_returned() -> None:
    failed = RequestCancelled(1, billed_attempts=(4,))
    assert charge_unanswered(lambda _u: 0.25, lambda: 10, failed) == 0.25


def test_no_other_module_builds_a_failed_calls_charge() -> None:
    """One concept, one home: outside the clients, which say which attempts may be
    billed, only `client/charge.py` reads `billed_attempts`. A second reader is a second
    charge, and the two would drift as `zeus/loop.py`'s and the scout's could."""
    readers: list[str] = []
    for path in sorted(SRC.rglob("*.py")):
        rel = path.relative_to(SRC).as_posix()
        if rel.startswith("client/"):
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        if any(
            isinstance(node, ast.Attribute) and node.attr == "billed_attempts"
            for node in ast.walk(tree)
        ):
            readers.append(rel)
    assert readers == []
