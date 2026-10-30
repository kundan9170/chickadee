"""D245's wire minor: a turn's files travel base64, a turn without any sends the body an older
server accepts, and the hello's limits are optional, so a client of an older server reads
`attach` as None and does not offer attaching."""

from __future__ import annotations

import base64
import json

from chickadee_serve.frames import WIRE_VERSION, Hello
from chickadee_wire import calls


def body(call: calls.RouteCall[calls.TurnAccepted]) -> dict[str, object]:
    raw = call.content()
    assert raw is not None
    return json.loads(raw)


def test_the_minor_that_carries_attachments() -> None:
    major, minor = (int(n) for n in WIRE_VERSION.split("."))
    assert (major, minor) >= (4, 14)


def test_a_turns_files_are_base64_and_absent_when_there_are_none() -> None:
    assert "attachments" not in body(calls.turn("S-1", "hello"))
    sent = body(calls.turn("S-1", "look", attachments=[("a.png", b"\x89PNG")]))
    assert sent["attachments"] == [
        {"name": "a.png", "data": base64.b64encode(b"\x89PNG").decode()}
    ]


def test_an_older_servers_hello_has_no_limits() -> None:
    hello = Hello.model_validate({"wire": "4.13", "engine": "0.0.0", "capabilities": []})
    assert hello.attach is None
