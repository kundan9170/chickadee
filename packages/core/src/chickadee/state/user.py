"""The person's own files beside their engine config (`~/.config/chickadee/`), for what a
frontend keeps between runs (the TUI's `tui.toml`): rendered by the frontend from the
person's own choice, never model-authored. Re-exported by the facade (`engine`), so a surface
reaches them through it alone."""

from __future__ import annotations

from pathlib import Path

from chickadee.durable.files import write_atomic
from chickadee.state.roots import USER_DIR, xdg_config_home

__all__ = ["user_file", "write_user_file"]


def user_file(name: str) -> Path:
    """`~/.config/chickadee/<name>`. Needs no engine: a preference is read before one exists."""
    return xdg_config_home() / USER_DIR / name


def write_user_file(path: Path, data: bytes) -> None:
    """`data` to the person's file at `path`, whole: a crash leaves the old or the new
    (`durable.write_atomic`), never a part. `OSError` when it cannot be written."""
    write_atomic(path, data)
