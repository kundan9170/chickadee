"""`/attach`: stages a file for the next line, lists, drops and clears, the REPL and
the TUI alike, and the next line sends what it holds."""

from __future__ import annotations

from pathlib import Path

import pytest
from engines import answer
from repl_driver import Typed, printed, repl

from chickadee_cli.commands import COMMANDS


def png(path: Path) -> Path:
    from PIL import Image

    Image.new("RGB", (30, 20), (9, 9, 9)).save(path, format="PNG")
    return path


def test_attach_is_a_command_the_repl_and_the_tui_share() -> None:
    assert COMMANDS["attach"].usage == "/attach [<path>|drop <n>|clear]"
    assert COMMANDS["attach"].on_the_wire


def test_attach_stages_lists_drops_and_sends(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    shot = png(tmp_path / "shot.png")
    notes = tmp_path / "notes.txt"
    notes.write_text("one\ntwo\n", encoding="utf-8")
    lines = [
        Typed(f"/attach {shot}"),
        Typed(f"/attach {notes}"),
        Typed(f"/attach {tmp_path / 'gone.txt'}"),
        Typed("/attach"),
        Typed("/attach drop 2"),
        Typed("/attach drop 9"),
        Typed("look at this"),
        Typed("/exit", after=printed("seen")),
    ]
    with repl(monkeypatch, [answer("seen")], lines) as (source, _code):
        # The REPL wraps to the terminal's width, so a long temp path can split a line;
        # whitespace folded to single spaces makes the words, not the wrap, what is asserted.
        out = " ".join(" ".join(source.out).split())
    assert "[1] shot.png · 30x20 · ~1 tok" in out
    assert "[2] notes.txt · 2 lines" in out
    assert "gone.txt: no such" in out, out
    assert "2 attached" in out
    assert "dropped notes.txt" in out
    assert "no attachment 9" in out
    assert "attached A-1 shot.png · image · 30x20" in out


def test_files_with_no_words_are_sent(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    shot = png(tmp_path / "shot.png")
    lines = [Typed(f"/attach {shot}"), Typed(""), Typed("/exit", after=printed("a picture"))]
    with repl(monkeypatch, [answer("a picture")], lines) as (source, _code):
        out = "\n".join(source.out)
    assert "attached A-1 shot.png" in out
