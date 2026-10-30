"""The composer attaches: a dropped file and `Ctrl+V` each
stage a chip, Enter sends them with the words, and a repository's text file dropped in is
mentioned rather than copied."""

from __future__ import annotations

import io
from collections.abc import Iterator
from pathlib import Path

import pytest
from engines import Step, answer
from hosting import SID, hosted
from textual.pilot import Pilot
from tui_frames import until

from chickadee_cli import attach
from chickadee_cli.tui.app import ChickadeeApp, _listed, connector
from chickadee_cli.tui.composer import HINT, Composer
from chickadee_cli.tui.feed import Feed
from chickadee_cli.tui.model import ZeusKind
from chickadee_wire.client import Client
from chickadee_wire.errors import Unanswered
from chickadee_wire.models import UserTurnP

pytestmark = pytest.mark.anyio

SIZE = (104, 33)


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture
def engine(monkeypatch: pytest.MonkeyPatch) -> Iterator[Client]:
    script: list[Step] = [answer("seen it")]
    with hosted(monkeypatch, script) as host, host.client() as client:
        yield client


def _app(client: Client) -> ChickadeeApp:
    return ChickadeeApp(
        feed=Feed(SID), connect=connector(client, SID), depth="truecolor", encoding="utf-8"
    )


def png_bytes() -> bytes:
    from PIL import Image

    out = io.BytesIO()
    Image.new("RGB", (40, 30), (5, 5, 5)).save(out, format="PNG")
    return out.getvalue()


def _sent(app: ChickadeeApp) -> list[UserTurnP]:
    return [
        e.record
        for e in app.state.zeus.entries
        if e.kind is ZeusKind.YOU and isinstance(e.record, UserTurnP)
    ]


def _chips(app: ChickadeeApp) -> str:
    return str(app.zeus.chips.render())


async def _send(app: ChickadeeApp, pilot: Pilot[None], text: str) -> None:
    app.zeus.composer.load_text(text)
    await pilot.press("enter")


async def test_a_dropped_file_is_a_chip_and_goes_with_the_words(
    engine: Client, tmp_path: Path
) -> None:
    shot = tmp_path / "shot.png"
    shot.write_bytes(png_bytes())
    app = _app(engine)
    async with app.run_test(size=SIZE) as pilot:
        app.zeus.composer.post_message(Composer.FileDropped(shot))
        await until(pilot, lambda: "[1] shot.png · 40x30" in _chips(app), "the chip")
        assert app.zeus.chips.has_class("-shown")
        await _send(app, pilot, "make it like this")
        await until(pilot, lambda: bool(_sent(app)), "the turn")
        (turn,) = _sent(app)
        assert [a.name for a in turn.attachments] == ["shot.png"]
        await until(pilot, lambda: not app.zeus.chips.has_class("-shown"), "the chips cleared")


@pytest.mark.parametrize(("maybe", "staged_again"), [(True, False), (False, True)])
async def test_a_turn_that_may_have_arrived_keeps_its_files_off_the_draft(
    engine: Client,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    maybe: bool,
    staged_again: bool,
) -> None:
    def unanswered(*_args: object, **_kwargs: object) -> object:
        raise Unanswered("POST", f"/session/{SID}/turn", "no answer", maybe_delivered=maybe)

    shot = tmp_path / "shot.png"
    shot.write_bytes(png_bytes())
    app = _app(engine)
    async with app.run_test(size=SIZE) as pilot:
        app.zeus.composer.post_message(Composer.FileDropped(shot))
        await until(pilot, lambda: "[1] shot.png" in _chips(app), "the chip")
        monkeypatch.setattr(engine, "turn", unanswered)
        await _send(app, pilot, "make it like this")

        def said() -> str:
            return "\n".join(
                e.text or "" for e in app.state.zeus.entries if e.kind is ZeusKind.OUTPUT
            )

        await until(pilot, lambda: "no answer" in said(), "the error")
        # Sent twice is a second copy recorded and paid for; never sent is lost if dropped.
        assert app.zeus.chips.has_class("-shown") is staged_again


async def test_ctrl_v_stages_the_clipboards_image(
    engine: Client, monkeypatch: pytest.MonkeyPatch
) -> None:
    image = png_bytes()
    monkeypatch.setattr(attach, "read_clipboard", lambda: image)
    app = _app(engine)
    async with app.run_test(size=SIZE) as pilot:
        app.zeus.composer.focus()
        await pilot.press("ctrl+v")
        await until(pilot, lambda: "[1] clipboard.png" in _chips(app), "the chip")


async def test_an_empty_clipboard_says_so_and_stages_nothing(
    engine: Client, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(attach, "read_clipboard", lambda: attach.NotStaged(attach.NO_IMAGE))
    app = _app(engine)
    async with app.run_test(size=SIZE) as pilot:
        app.zeus.composer.focus()
        await pilot.press("ctrl+v")

        def said() -> str:
            return "\n".join(
                e.text or "" for e in app.state.zeus.entries if e.kind is ZeusKind.OUTPUT
            )

        await until(pilot, lambda: "the clipboard holds no image" in said(), "the refusal")
        assert not app.zeus.chips.has_class("-shown")


def test_a_repository_file_is_mentioned_by_its_listed_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.py").write_text("x = 1\n")
    assert _listed(tmp_path / "src" / "app.py", ["src/app.py"]) == "src/app.py"
    assert _listed(tmp_path / "src" / "app.py", ["other.py"]) is None
    assert _listed(Path("/etc/hostname"), ["src/app.py"]) is None


def test_the_hint_names_ctrl_v() -> None:
    assert HINT.endswith("^V image")
