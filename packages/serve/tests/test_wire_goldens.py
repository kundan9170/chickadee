"""The wire contract is pinned, and a change to it moves `WIRE_VERSION` (`golden.py`)."""

from __future__ import annotations

import json

from chickadee_serve import frames, golden


def test_the_committed_golden_is_the_contract() -> None:
    committed = golden.GOLDEN.read_text(encoding="utf-8")
    assert committed == golden.render(), (
        "the wire contract changed. If it should, bump WIRE_VERSION in frames.py (minor for an "
        "addition, major for a removal or change) and run `python -m chickadee_serve.golden`"
    )


def test_the_version_names_this_contract() -> None:
    """A contract changed without a bump keeps the old version's sha in `versions.json`, which
    no longer matches; a bump records the new one beside it."""
    versions = json.loads(golden.VERSIONS.read_text(encoding="utf-8"))
    doc = json.loads(golden.GOLDEN.read_text(encoding="utf-8"))
    assert doc["wire"] == frames.WIRE_VERSION
    assert versions.get(frames.WIRE_VERSION) == golden.content_sha(doc)


def test_every_session_kind_is_described_or_marked_opaque() -> None:
    from chickadee.context.vocabulary import SESSION_KINDS

    records = golden.contract()["session_records"]
    assert set(records) == set(SESSION_KINDS)
    assert records["question_asked"].get("x-opaque") is None, "a client renders questions"
    assert records["turn"].get("x-opaque") is None


def test_a_major_bump_is_what_a_client_refuses() -> None:
    assert frames.major("1.0") == 1
    assert frames.major("2.3") == 2
