"""D245 over the wire: a turn carries its attachments as base64, the engine checks each again,
a bad one is a 422 that names it with nothing recorded, and the hello says what a turn may
carry."""

from __future__ import annotations

import base64
import dataclasses
import io
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import cast

import pytest
from engines import answer, config_for, raw_config, start, with_sessions
from repos import init_repo
from serve_helpers import Live, served_engine

from chickadee.context.attachment import AttachCaps
from chickadee_serve.frames import AttachLimits

SID = "S-attach"


def b64(data: bytes) -> str:
    return base64.b64encode(data).decode()


def png() -> bytes:
    from PIL import Image

    out = io.BytesIO()
    Image.new("RGB", (40, 20), (1, 2, 3)).save(out, format="PNG")
    return out.getvalue()


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    return with_sessions(init_repo(tmp_path / "repo", commit=None), SID)


@contextmanager
def live(monkeypatch: pytest.MonkeyPatch, repo: Path) -> Iterator[Live]:
    engine, _ = start(monkeypatch, repo, config_for(raw_config()), [answer("seen")])
    try:
        with served_engine(engine) as h:
            yield h
    finally:
        engine.close()


def test_a_turn_carries_its_files_and_the_record_names_them(
    monkeypatch: pytest.MonkeyPatch, repo: Path
) -> None:
    with live(monkeypatch, repo) as h:
        body = {
            "text": "look at these",
            "attachments": [
                {"name": "shot.png", "data": b64(png())},
                {"name": "crash.log", "data": b64(b"boom\n")},
            ],
        }
        r = h.client.post(f"/session/{SID}/turn", json=body, headers=h.headers())
        assert r.status_code == 202, r.text
        turn = h.until(SID, "turn")
    attached = cast("list[dict[str, object]]", turn.payload["attachments"])
    assert [(a["id"], a["kind"], a["name"]) for a in attached] == [
        ("A-1", "image", "shot.png"),
        ("A-2", "text", "crash.log"),
    ]


@pytest.mark.parametrize(
    ("attachment", "said"),
    [
        (
            {"name": "x.bin", "data": b64(bytes(range(256)))},
            "attachment 1 (x.bin): not an image",
        ),
        (
            {"name": "x.png", "data": "not base64!"},
            "attachment 1 (x.png): its data is not base64",
        ),
    ],
)
def test_a_bad_file_refuses_the_turn_whole_and_records_nothing(
    monkeypatch: pytest.MonkeyPatch, repo: Path, attachment: dict[str, str], said: str
) -> None:
    with live(monkeypatch, repo) as h:
        body = {"text": "look", "attachments": [attachment]}
        r = h.client.post(f"/session/{SID}/turn", json=body, headers=h.headers())
        assert r.status_code == 422, r.text
        assert r.json()["code"] == "BAD_REQUEST"
        assert said in r.json()["message"]
        assert [f for f in h.journal(SID) if f.kind == "turn"] == []


def test_the_hello_says_what_a_turn_may_carry(
    monkeypatch: pytest.MonkeyPatch, repo: Path
) -> None:
    with live(monkeypatch, repo) as h:
        hello = h.client.get("/").json()
    assert hello["attach"] == dataclasses.asdict(AttachCaps.defaults())


def test_the_wire_limits_are_the_engines_field_for_field() -> None:
    """`AttachLimits` is the core-free copy of `AttachCaps`: the two may not drift."""
    wire = set(AttachLimits.model_fields)
    core = {f.name for f in dataclasses.fields(AttachCaps)}
    assert wire == core
