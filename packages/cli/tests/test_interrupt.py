"""Ctrl-C levels: the first press stops a turn at its next boundary; a
second within the window stops it now; a press after the window is a first press again."""

from __future__ import annotations

import json
from pathlib import Path

import httpx

from chickadee_cli.interrupt import HARD_WINDOW_S, Interrupter
from chickadee_serve.bind import Uds
from chickadee_wire.client import Client


def test_one_press_is_a_boundary_stop_and_two_are_a_hard_one() -> None:
    sent: list[bool] = []

    def record(request: httpx.Request) -> httpx.Response:
        sent.append(json.loads(request.content)["hard"])
        return httpx.Response(202, json={})

    now = [100.0]
    client = Client(Uds(Path("/x.sock")), "t", transport=httpx.MockTransport(record))
    presses = Interrupter(client, "S", clock=lambda: now[0])
    assert presses.press() == "boundary"
    now[0] += HARD_WINDOW_S / 2
    assert presses.press() == "hard"
    now[0] += 1.0
    assert presses.press() == "boundary", "a hard stop starts the count again"
    now[0] += HARD_WINDOW_S * 2
    assert presses.press() == "boundary", "too late to be the second press"
    assert sent == [False, True, False, False]
