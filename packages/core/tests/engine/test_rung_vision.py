"""Every rung's reader sees as its model does.

`supports_images` is the model's (`[models.<name>]`), and `EngineConfig.vision_for(rung)` is
the one way a reader's sight is built: Zeus's reader, through `PacketEngine.turn()`, and the
scout's (`tests/zeus/test_locate.py`). Two rungs naming one model see alike, and moving a rung
to another model moves its sight.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import pytest
from PIL import Image
from repos import init_repo
from zeus_turn_helpers import SID, WAIT_S, call, calls, config_for, result, start, with_sessions

from chickadee.client.request import ImagePart
from chickadee.config import EngineConfig
from chickadee.engine import PacketEngine


@pytest.fixture(autouse=True)
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "home"
    path.mkdir()
    monkeypatch.setenv("HOME", str(path))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    return path


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    path = with_sessions(init_repo(tmp_path / "repo", commit=None), SID)
    image = io.BytesIO()
    Image.new("RGB", (64, 48), (200, 30, 30)).save(image, format="PNG")
    (path / "shot.png").write_bytes(image.getvalue())
    return path


def _zeus_model(raw: dict[str, Any]) -> dict[str, Any]:
    return raw["models"][raw["rungs"]["high-param"]]


@pytest.mark.parametrize("sees", [True, False])
def test_zeus_reads_an_image_as_its_rungs_model_can(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    raw: dict[str, Any],
    engines: list[PacketEngine],
    sees: bool,
) -> None:
    _zeus_model(raw)["supports_images"] = sees
    script = [calls(call("read", "r1", path="shot.png", depth="full"))]
    engine, model = start(monkeypatch, repo, config_for(raw), script)
    engines.append(engine)
    engine.turn(SID, "how does the page look?").wait(timeout=WAIT_S)

    answered = result(engine, "r1")
    refused = "NOT_SUPPORTED{capability: images, rung: high-param}" in answered
    assert refused is not sees
    sent = [
        part
        for m in model.requests[-1].messages
        if not isinstance(m.content, str)
        for part in m.content
        if isinstance(part, ImagePart)
    ]
    assert len(sent) == (1 if sees else 0)


def _config(rungs: dict[str, str], blind: tuple[str, ...]) -> EngineConfig:
    model = {"base_url": "http://x", "model_id": "m", "context_window": 65_536}
    model["max_output_tokens"] = 16_384
    return EngineConfig.with_defaults(
        {
            "models": {
                name: model | ({"supports_images": False} if name in blind else {})
                for name in ("seeing", "also-seeing", "blind")
            },
            "rungs": rungs,
            "spend": {"hard_cost_usd": 5.0},
        }
    )


def test_two_rungs_naming_one_model_see_alike() -> None:
    config = _config(
        {"high-param": "blind", "mid-param": "blind", "low-param": "seeing"}, ("blind",)
    )
    high, mid = config.vision_for("high-param"), config.vision_for("mid-param")
    assert (high.supports_images, mid.supports_images) == (False, False)
    assert (high.rung, mid.rung) == ("high-param", "mid-param")
    assert config.vision_for("low-param").supports_images


def test_moving_a_rung_to_another_model_moves_its_sight() -> None:
    rest = {"mid-param": "also-seeing", "low-param": "seeing"}
    before = _config({"high-param": "seeing", **rest}, ("blind",))
    after = _config({"high-param": "blind", **rest}, ("blind",))
    assert before.vision_for("high-param").supports_images
    assert not after.vision_for("high-param").supports_images
