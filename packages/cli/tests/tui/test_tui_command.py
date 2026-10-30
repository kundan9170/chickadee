"""`chickadee tui`: the screen over an engine
hosted in this process, or over the server in `--server`'s directory.

What these pin: with no terminal it starts no screen and names the batch commands, as bare
`chickadee` does (exit 6); `--server` with nothing listening there says so (exit 1); and with a
replay's directory it opens the replay's session, the screen given the repository's name only
when it hosts the engine. `test_command_surface.py` holds its row to §6a's.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
import typer
from recordings import recording, replayed

from chickadee_cli import main
from chickadee_cli.surface import NOT_A_TTY, USAGE


def test_with_no_terminal_it_starts_no_screen_and_names_the_batch_commands(
    tmp_path: Path,
) -> None:
    done = subprocess.run(
        [sys.executable, "-c", "from chickadee_cli.main import main; main()", "tui"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
        stdin=subprocess.DEVNULL,
    )
    assert done.returncode == NOT_A_TTY, done.stderr
    assert "chickadee tui: not a terminal, so no TUI." in done.stderr
    assert 'chickadee run --yes "<goal>"' in done.stderr


def test_bare_chickadee_still_says_what_it_said() -> None:
    assert main.BATCH_EQUIVALENT == (
        "chickadee: not a terminal, so no REPL. For a script, say the goal as a command:\n"
        '  chickadee run --yes "<goal>"     or     chickadee ask "<question>"'
    ), "byte for byte, the message before the TUI shared its hint"


def test_a_server_directory_with_nothing_listening_says_so(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(main, "_screen", lambda: True)
    with pytest.raises(typer.Exit) as left:
        main.tui(server=tmp_path)
    assert left.value.exit_code == USAGE
    assert "chickadee tui: no server answers at" in capsys.readouterr().err


def test_a_replay_s_directory_opens_its_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    opened: list[tuple[str, str | None, bool, bool]] = []

    def screen(
        _client: object,
        session_id: str,
        *,
        repo: str | None,
        resume_paused: bool,
        full_screen: bool,
        remember: object,
    ) -> int:
        opened.append((session_id, repo, resume_paused, full_screen))
        return 0

    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.setattr(main, "_screen", lambda: True)
    monkeypatch.setattr("chickadee_cli.tui.app.run", screen)
    with (
        replayed(recording(tmp_path / "recording")) as served,
        pytest.raises(typer.Exit) as left,
    ):
        main.tui(server=served.directory, resume_latest=True)
    assert left.value.exit_code == 0
    # A replay's screen names no repository; `--continue` asks for a paused run to go on, which
    # the screen declines for a replay itself (`Talk`), as every write.
    assert opened == [(served.session_id, None, True, False)], "no tui.toml: not full screen"
