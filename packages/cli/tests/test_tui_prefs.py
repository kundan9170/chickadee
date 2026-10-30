"""The TUI's preferences file (`chickadee_cli/tui_prefs.py`): what `chickadee tui` reads
before the screen opens and writes when the person toggles full screen. A file that is missing
or says nothing this module wrote reads as the defaults, never an error."""

from __future__ import annotations

from pathlib import Path

import pytest

from chickadee_cli import tui_prefs
from chickadee_cli.tui_prefs import TuiPrefs, load, save


def test_it_lives_beside_the_user_s_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    assert tui_prefs.prefs_path() == tmp_path / "chickadee" / "tui.toml"


def test_with_no_home_to_find_it_in_there_is_none(monkeypatch: pytest.MonkeyPatch) -> None:
    """No `HOME` and no passwd entry: `Path.home()` raises; the screen opens with the defaults
    and keeps nothing, never failing to open."""

    def homeless(_name: str) -> Path:
        raise RuntimeError("Could not determine home directory.")

    monkeypatch.setattr(tui_prefs, "user_file", homeless)
    assert tui_prefs.prefs_path() is None


def test_what_is_saved_is_read_back(tmp_path: Path) -> None:
    path = tmp_path / "chickadee" / "tui.toml"
    assert load(path) == TuiPrefs(), "missing: the defaults"
    save(path, TuiPrefs(full_screen=True))
    assert load(path) == TuiPrefs(full_screen=True)
    assert "full_screen = true" in path.read_text(encoding="utf-8")
    save(path, TuiPrefs(full_screen=False))
    assert load(path) == TuiPrefs(full_screen=False)


@pytest.mark.parametrize("text", ["full_screen = [", 'full_screen = "yes"', "\xff"])
def test_a_file_it_did_not_write_reads_as_the_defaults(tmp_path: Path, text: str) -> None:
    path = tmp_path / "tui.toml"
    path.write_bytes(text.encode("latin-1"))
    assert load(path) == TuiPrefs()
