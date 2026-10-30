"""A client outlives its server: the server restarts and issues new
tokens, and a client that read its token from the server's file reads the new one on the
first 401, while a stream reconnects from its last seq and misses nothing."""

from __future__ import annotations

import pytest
from engines import answer
from hosting import SID, answered, repository, scripted, short_root

from chickadee_serve.frames import RecordFrame, TokenClass
from chickadee_wire.client import Client
from chickadee_wire.errors import WireError
from chickadee_wire.events import Events
from chickadee_wire.host import host_in_process
from chickadee_wire.transport import token_in


def _through(events: Events, seq: int) -> list[int]:
    """The record seqs `events` yields, up to and including `seq`."""
    seen: list[int] = []
    while not seen or seen[-1] < seq:
        frame = next(events)
        if isinstance(frame, RecordFrame):
            seen.append(frame.seq)
    return seen


def test_a_client_and_its_stream_carry_on_across_a_restart(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = repository(monkeypatch, short_root())
    scripted(monkeypatch, [answer("one"), answer("two")])
    first = host_in_process(repo)
    with Client.attach(first.state_dir) as client:
        client.turn(SID, "before")
        answered(client, 1)
        before = client.session_journal(SID)[-1].seq
        events = client.session_events(SID)
        assert _through(events, before) == list(range(1, before + 1))
        old = token_in(first.state_dir)
        first.close()
        with host_in_process(repo) as second:
            assert token_in(second.state_dir) != old, "a restart issues new tokens"
            client.turn(SID, "after")  # a 401 with the old token, then the new one
            answered(client, 2)
            after = client.session_journal(SID)[-1].seq
            assert _through(events, after) == list(range(before + 1, after + 1))
            events.close()


def test_a_fixed_token_does_not_survive_a_restart(monkeypatch: pytest.MonkeyPatch) -> None:
    repo = repository(monkeypatch, short_root())
    scripted(monkeypatch, [])
    first = host_in_process(repo)
    stale = first.token(TokenClass.INTERACTIVE)
    address = first.address
    first.close()
    with (
        host_in_process(repo),
        Client(address, stale) as client,
        pytest.raises(WireError) as refused,
    ):
        client.sessions()
    assert refused.value.status == 401
