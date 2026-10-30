"""The TUI's own preferences: `$XDG_CONFIG_HOME/chickadee/tui.toml`, the person's, kept
between screens. Today one: whether the header opens as one line (`full_screen`, `^F`).

The screen never writes a file (it reaches no further than the wire): `chickadee tui`
reads this before it opens the screen and hands the screen a `remember` that writes it, through
the engine's facade (`engine.write_user_file`, the storage module's atomic write), so a crash
leaves the old file or the new.

**A preference never stops the screen opening.** A file that is missing, unreadable or not
what this module writes reads as the defaults; the next choice writes it whole again.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from chickadee.engine import user_file, write_user_file

__all__ = ["TUI_PREFS", "TuiPrefs", "load", "prefs_path", "save"]

TUI_PREFS: Final = "tui.toml"


@dataclass(frozen=True)
class TuiPrefs:
    full_screen: bool = False
    """The header as one line, no bird (`^F`, `/fullscreen`)."""


def prefs_path() -> Path | None:
    """`$XDG_CONFIG_HOME/chickadee/tui.toml`, beside the user's `config.toml`; None where there
    is no home to find it in (no `HOME`, no passwd entry): the defaults, and nothing kept."""
    try:
        return user_file(TUI_PREFS)
    except RuntimeError:
        return None


def load(path: Path) -> TuiPrefs:
    """What `path` says, or the defaults where it says nothing readable."""
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError):
        return TuiPrefs()
    full_screen = data.get("full_screen")
    return TuiPrefs(full_screen=full_screen if isinstance(full_screen, bool) else False)


def save(path: Path, prefs: TuiPrefs) -> None:
    """`prefs` to `path`, whole, atomically."""
    text = (
        "# The TUI's preferences, written by `chickadee tui` (^F toggles full_screen).\n"
        f"full_screen = {'true' if prefs.full_screen else 'false'}\n"
    )
    write_user_file(path, text.encode("utf-8"))
